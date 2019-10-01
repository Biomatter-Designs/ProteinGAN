from common.bio.constants import SMILES_CHARACTER_TO_ID, ID_TO_SMILES_CHARACTER


def from_smiles_to_id(data, column):
    """Converts sequences from smiles to ids

    Args:
      data: data that contains characters that need to be converted to ids
      column: a column of the dataframe that contains characters that need to be converted to ids

    Returns:
      array of ids

    """
    return [[SMILES_CHARACTER_TO_ID[char] for char in val] for index, val in data[column].iteritems()]


def from_id_from_smiles(data, column):
    """Converts sequences from ids to smiles characters

    Args:
      data: data that contains ids that need to be converted to characters
      column: a column of the dataframe that contains ids that need to be converted to characters

    Returns:
      array of characters

    """
    return [[ID_TO_SMILES_CHARACTER[id] for id in val] for index, val in data[column].iteritems()]



