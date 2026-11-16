from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd
STANDARD_COLUMNS = ['drug_id', 'target_id', 'smiles', 'protein_sequence', 'label']
DATA1_COLUMNS = {'SMILES', 'Protein', 'Y'}

def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode('utf-8')).hexdigest()[:16]
    return f'{prefix}_{digest}'

def normalize_dti_dataframe(df: pd.DataFrame, source: str | Path='<dataframe>') -> pd.DataFrame:
    if set(STANDARD_COLUMNS).issubset(df.columns):
        return df[STANDARD_COLUMNS].copy()
    if DATA1_COLUMNS.issubset(df.columns):
        smiles = df['SMILES'].astype(str)
        proteins = df['Protein'].astype(str)
        return pd.DataFrame({'drug_id': smiles.map(lambda value: stable_id('D', value)), 'target_id': proteins.map(lambda value: stable_id('T', value)), 'smiles': smiles, 'protein_sequence': proteins, 'label': df['Y'].astype(float)})
    missing_standard = sorted(set(STANDARD_COLUMNS) - set(df.columns))
    missing_data1 = sorted(DATA1_COLUMNS - set(df.columns))
    raise ValueError(f'{source} does not match a supported DTI CSV schema. Missing standard columns: {missing_standard}; missing data1 columns: {missing_data1}')

def read_dti_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_dti_dataframe(df, source=path)
import torch
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_PAD_INDEX = 0
AA_UNK_INDEX = 21
AA_VOCAB_SIZE = 22
AA_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}
SMILES_CHARACTERS = ''.join((chr(code) for code in range(33, 127)))
SMILES_TO_INDEX = {character: index + 2 for index, character in enumerate(SMILES_CHARACTERS)}
SMILES_PAD_INDEX = 0
SMILES_UNK_INDEX = 1
SMILES_VOCAB_SIZE = len(SMILES_CHARACTERS) + 2

def encode_amino_acids(sequence: str, max_length: int) -> torch.Tensor:
    indices = [AA_TO_INDEX.get(residue.upper(), AA_UNK_INDEX) for residue in sequence[:max_length]]
    return torch.tensor(indices or [AA_UNK_INDEX], dtype=torch.long)

def encode_smiles_characters(smiles: str, max_length: int) -> torch.Tensor:
    indices = [SMILES_TO_INDEX.get(character, SMILES_UNK_INDEX) for character in smiles[:max_length]]
    return torch.tensor(indices or [SMILES_UNK_INDEX], dtype=torch.long)
import torch
ATOM_FEATURE_DIM = 128
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AMINO_ACID_TO_INDEX = {amino_acid: index + 1 for index, amino_acid in enumerate(AMINO_ACIDS)}

def encode_protein_sequence(sequence: str, max_length: int) -> torch.Tensor:
    """Encode a protein sequence with 0 reserved for padding/unknown residues."""
    indices = [AMINO_ACID_TO_INDEX.get(residue, 0) for residue in sequence[:max_length]]
    return torch.tensor(indices or [0], dtype=torch.long)

def smiles_to_graph(smiles: str, max_atoms: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact atom and bond tensors from a SMILES string.

    RDKit is intentionally imported only here so embedding-only experiments do
    not require it. Atom features are a stable, fixed-size encoding and bond
    features use four bond categories plus aromaticity.
    """
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError('RDKit is required for model.use_multiview_residual=true. Install it with `conda install -c conda-forge rdkit`.') from exc
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f'Invalid SMILES encountered: {smiles!r}')
    atom_count = min(molecule.GetNumAtoms(), max_atoms)
    node_features = torch.zeros((atom_count, ATOM_FEATURE_DIM), dtype=torch.float32)
    adjacency = torch.zeros((atom_count, atom_count), dtype=torch.bool)
    edge_features = torch.zeros((atom_count, atom_count, 5), dtype=torch.float32)
    for atom_index, atom in enumerate(molecule.GetAtoms()):
        if atom_index >= atom_count:
            break
        atomic_number = min(max(atom.GetAtomicNum(), 1), 118)
        node_features[atom_index, atomic_number - 1] = 1.0
        node_features[atom_index, 118] = atom.GetDegree() / 4.0
        node_features[atom_index, 119] = atom.GetFormalCharge() / 4.0
        node_features[atom_index, 120] = float(atom.GetIsAromatic())
        node_features[atom_index, 121] = atom.GetTotalNumHs() / 4.0
        node_features[atom_index, 122] = float(atom.IsInRing())
    bond_to_index = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
    for bond in molecule.GetBonds():
        source, target = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if source >= atom_count or target >= atom_count:
            continue
        bond_features = torch.zeros(5, dtype=torch.float32)
        bond_features[bond_to_index.get(bond.GetBondType(), 0)] = 1.0
        bond_features[4] = float(bond.GetIsAromatic())
        adjacency[source, target] = True
        adjacency[target, source] = True
        edge_features[source, target] = bond_features
        edge_features[target, source] = bond_features
    return (node_features, adjacency, edge_features)
from pathlib import Path
import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Dataset backed by precomputed per-drug and per-target token embeddings."""

    def __init__(self, csv_path: str | Path, drug_embedding_dir: str | Path, protein_embedding_dir: str | Path, max_drug_len: int=128, max_protein_len: int=1024, use_multiview_residual: bool=False, max_graph_atoms: int=128, max_raw_protein_len: int=1024, drug_encoder_type: str='plm', protein_encoder_type: str='plm') -> None:
        self.df = read_dti_csv(csv_path)
        self.drug_embedding_dir = Path(drug_embedding_dir)
        self.protein_embedding_dir = Path(protein_embedding_dir)
        self.max_drug_len = max_drug_len
        self.max_protein_len = max_protein_len
        self.use_multiview_residual = use_multiview_residual
        self.max_graph_atoms = max_graph_atoms
        self.max_raw_protein_len = max_raw_protein_len
        self.drug_encoder_type = drug_encoder_type
        self.protein_encoder_type = protein_encoder_type
        self._graph_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._smiles_token_cache: dict[str, torch.Tensor] = {}
        self._protein_token_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        drug_id = str(row['drug_id'])
        target_id = str(row['target_id'])
        drug = torch.load(self.drug_embedding_dir / f'{drug_id}.pt', map_location='cpu') if self.drug_encoder_type == 'plm' else torch.zeros((1, 1), dtype=torch.float32)
        protein = torch.load(self.protein_embedding_dir / f'{target_id}.pt', map_location='cpu') if self.protein_encoder_type == 'plm' else torch.zeros((1, 1), dtype=torch.float32)
        drug = drug[:self.max_drug_len].float()
        protein = protein[:self.max_protein_len].float()
        label = torch.tensor(float(row['label']), dtype=torch.float32)
        item: dict[str, torch.Tensor | str] = {'drug_id': drug_id, 'target_id': target_id, 'drug_embedding': drug, 'protein_embedding': protein, 'label': label}
        if self.drug_encoder_type == 'smiles_cnn':
            if drug_id not in self._smiles_token_cache:
                self._smiles_token_cache[drug_id] = encode_smiles_characters(str(row['smiles']), self.max_drug_len)
            item['raw_smiles_tokens'] = self._smiles_token_cache[drug_id]
        if self.protein_encoder_type == 'aa_cnn':
            if target_id not in self._protein_token_cache:
                self._protein_token_cache[target_id] = encode_amino_acids(str(row['protein_sequence']), self.max_protein_len)
            item['raw_protein_tokens'] = self._protein_token_cache[target_id]
        if self.use_multiview_residual:
            smiles = str(row['smiles'])
            if drug_id not in self._graph_cache:
                self._graph_cache[drug_id] = smiles_to_graph(smiles, self.max_graph_atoms)
            node_features, adjacency, edge_features = self._graph_cache[drug_id]
            item.update({'graph_node_features': node_features, 'graph_adjacency': adjacency, 'graph_edge_features': edge_features, 'protein_sequence': encode_protein_sequence(str(row['protein_sequence']), self.max_raw_protein_len)})
        return item
import torch

def _pad_tokens(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(tensors)
    max_len = max((t.size(0) for t in tensors))
    dim = tensors[0].size(-1)
    padded = tensors[0].new_zeros((batch, max_len, dim))
    mask = torch.ones((batch, max_len), dtype=torch.bool)
    for i, tensor in enumerate(tensors):
        length = tensor.size(0)
        padded[i, :length] = tensor
        mask[i, :length] = False
    return (padded, mask)

def _pad_indices(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch = len(tensors)
    max_len = max((tensor.size(0) for tensor in tensors))
    padded = torch.zeros((batch, max_len), dtype=torch.long)
    mask = torch.ones((batch, max_len), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        length = tensor.size(0)
        padded[index, :length] = tensor
        mask[index, :length] = False
    return (padded, mask)

def _pad_graphs(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_nodes = max((item['graph_node_features'].size(0) for item in batch))
    batch_size = len(batch)
    node_dim = batch[0]['graph_node_features'].size(-1)
    edge_dim = batch[0]['graph_edge_features'].size(-1)
    nodes = torch.zeros((batch_size, max_nodes, node_dim), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, max_nodes, max_nodes), dtype=torch.bool)
    edges = torch.zeros((batch_size, max_nodes, max_nodes, edge_dim), dtype=torch.float32)
    node_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item['graph_node_features'].size(0)
        nodes[index, :count] = item['graph_node_features']
        adjacency[index, :count, :count] = item['graph_adjacency']
        edges[index, :count, :count] = item['graph_edge_features']
        node_mask[index, :count] = False
    return {'graph_node_features': nodes, 'graph_adjacency': adjacency, 'graph_edge_features': edges, 'graph_node_mask': node_mask}

def embedding_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    drug, drug_mask = _pad_tokens([item['drug_embedding'] for item in batch])
    protein, protein_mask = _pad_tokens([item['protein_embedding'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    result: dict[str, torch.Tensor | list[str]] = {'drug_ids': [item['drug_id'] for item in batch], 'target_ids': [item['target_id'] for item in batch], 'drug_embedding': drug, 'protein_embedding': protein, 'drug_mask': drug_mask, 'protein_mask': protein_mask, 'label': labels}
    if 'raw_smiles_tokens' in batch[0]:
        raw_smiles, raw_smiles_mask = _pad_indices([item['raw_smiles_tokens'] for item in batch])
        result['raw_smiles_tokens'] = raw_smiles
        result['raw_smiles_mask'] = raw_smiles_mask
    if 'raw_protein_tokens' in batch[0]:
        raw_protein, raw_protein_mask = _pad_indices([item['raw_protein_tokens'] for item in batch])
        result['raw_protein_tokens'] = raw_protein
        result['raw_protein_mask'] = raw_protein_mask
    if 'graph_node_features' in batch[0]:
        result.update(_pad_graphs(batch))
        protein_sequence, protein_sequence_mask = _pad_tokens([item['protein_sequence'].unsqueeze(-1).float() for item in batch])
        result['protein_sequence'] = protein_sequence.squeeze(-1).long()
        result['protein_sequence_mask'] = protein_sequence_mask
    return result
import argparse
from pathlib import Path
import pandas as pd
SPLIT_FILE_MAP = {'train': ('train.csv', 'train.csv'), 'validation': ('validation.csv', 'valid.csv'), 'test': ('test.csv', 'test.csv')}

def make_ids(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Create entity IDs from content, never from clustering assignments."""
    drug_ids = df['SMILES'].astype(str).map(lambda value: stable_id('D', value))
    target_ids = df['Protein'].astype(str).map(lambda value: stable_id('T', value))
    return (drug_ids, target_ids)

def convert_frame(df: pd.DataFrame, source: str | Path) -> tuple[pd.DataFrame, int]:
    required = {'SMILES', 'Protein', 'Y'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{source} is missing required columns: {missing}')
    drug_ids, target_ids = make_ids(df)
    out = pd.DataFrame({'drug_id': drug_ids, 'target_id': target_ids, 'smiles': df['SMILES'].astype(str), 'protein_sequence': df['Protein'].astype(str), 'label': pd.to_numeric(df['Y'], errors='raise').astype(float)})
    if out[['smiles', 'protein_sequence', 'label']].isna().any().any():
        raise ValueError(f'{source} contains missing SMILES, protein sequences, or labels')
    labels = set(out['label'].unique())
    if not labels.issubset({0.0, 1.0}):
        raise ValueError(f'{source} contains non-binary labels: {sorted(labels)}')
    # Preserve the original row order and multiplicity.
    return (out, 0)

def convert_file(input_path: Path, output_path: Path, id_strategy: str='hash') -> tuple[int, int, int]:
    if id_strategy != 'hash':
        raise ValueError('Only stable hash IDs are supported')
    df = pd.read_csv(input_path)
    out, _ = convert_frame(df, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return (len(out), out['drug_id'].nunique(), out['target_id'].nunique())

def validate_cold_split(scenario: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    shared_drugs = set(train['drug_id']) & set(test['drug_id'])
    shared_targets = set(train['target_id']) & set(test['target_id'])
    if scenario == 'e2' and shared_drugs:
        raise ValueError(f'E2 split has {len(shared_drugs)} test drugs present in training')
    if scenario == 'e3' and shared_targets:
        raise ValueError(f'E3 split has {len(shared_targets)} test targets present in training')
    if scenario == 'e4' and (shared_drugs or shared_targets):
        raise ValueError(f'E4 split violates double-cold separation: shared_drugs={len(shared_drugs)}, shared_targets={len(shared_targets)}')

def convert_dataset(input_root: Path, output_root: Path, scenarios: list[str], folds: list[str] | None) -> None:
    rows = []
    for scenario in scenarios:
        scenario_dir = input_root / scenario
        if not scenario_dir.exists():
            raise FileNotFoundError(f'Scenario directory not found: {scenario_dir}')
        if folds is None:
            fold_dirs = sorted([path for path in scenario_dir.iterdir() if path.is_dir()], key=lambda path: int(path.name) if path.name.isdigit() else path.name)
        else:
            fold_dirs = [scenario_dir / fold for fold in folds]
        if not fold_dirs:
            raise FileNotFoundError(f'No fold directories found in {scenario_dir}')
        for fold_dir in fold_dirs:
            if not fold_dir.exists():
                raise FileNotFoundError(f'Fold directory not found: {fold_dir}')
            converted: dict[str, pd.DataFrame] = {}
            duplicate_counts: dict[str, int] = {}
            for source_name, (input_name, target_name) in SPLIT_FILE_MAP.items():
                input_path = fold_dir / input_name
                if source_name == 'validation' and (not input_path.exists()):
                    input_path = fold_dir / 'valid.csv'
                if not input_path.exists():
                    raise FileNotFoundError(f'Split file not found: {input_path}')
                converted[source_name], duplicate_counts[source_name] = convert_frame(pd.read_csv(input_path), input_path)
            cross_removed = {'train': 0, 'validation': 0, 'test': 0}
            validate_cold_split(scenario, converted['train'], converted['test'])
            for source_name, (_, target_name) in SPLIT_FILE_MAP.items():
                output_path = output_root / scenario / fold_dir.name / target_name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frame = converted[source_name]
                frame.to_csv(output_path, index=False)
                rows.append({'scenario': scenario, 'fold': fold_dir.name, 'split': target_name.removesuffix('.csv'), 'rows': len(frame), 'drugs': frame['drug_id'].nunique(), 'targets': frame['target_id'].nunique(), 'within_split_duplicates_removed': duplicate_counts[source_name], 'cross_split_duplicates_removed': cross_removed[source_name], 'id_strategy': 'sha1_content_hash', 'path': str(output_path).replace('\\', '/')})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / 'conversion_summary.csv', index=False)
    print(f'Converted DTI splits to {output_root}')
    print(summary[['scenario', 'fold', 'split', 'rows', 'drugs', 'targets', 'within_split_duplicates_removed', 'cross_split_duplicates_removed']].to_string(index=False))

def convert_cli() -> None:
    parser = argparse.ArgumentParser(description='Convert random/E2/E3/E4 DTI splits to the PriorSL-DTI schema.')
    parser.add_argument('--input_root', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--scenarios', nargs='+', default=['random', 'e2', 'e3', 'e4'])
    parser.add_argument('--folds', nargs='+', default=None)
    args = parser.parse_args()
    convert_dataset(input_root=Path(args.input_root), output_root=Path(args.output_root), scenarios=args.scenarios, folds=args.folds)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] != "convert":
        raise SystemExit("Usage: python data.py convert [arguments]")
    sys.argv.pop(1)
    convert_cli()
