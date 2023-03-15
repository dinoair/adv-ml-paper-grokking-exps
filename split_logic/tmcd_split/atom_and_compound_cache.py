import json
import os

from split_logic.grammar.sparql_parser import SPARQLParser


class AtomAndCompoundCache:

    def __init__(self, parser: SPARQLParser, query_key_name):
        self.query_parser = parser
        self.query_to_atoms = {}
        self.query_to_compounds = {}
        self.query_key_name = query_key_name

    def get_atoms(self, query_dict):
        query = query_dict[self.query_key_name]
        if query not in self.query_to_atoms:
            atoms = self.query_parser.get_atoms(query)
            self.query_to_atoms[query] = atoms
        else:
            atoms = self.query_to_atoms[query]
        return atoms

    def extract_compounds_list(self, compound_dict):
        compound_list = []
        for compound_name in compound_dict:
            compound_list += compound_dict[compound_name]
        return compound_list

    def get_compounds(self, query_dict):
        query = query_dict[self.query_key_name]
        if query not in self.query_to_compounds:
            compounds_dict = self.query_parser.get_compounds(query)
            self.query_to_compounds[query] = compounds_dict
        else:
            compounds_dict = self.query_to_compounds[query]
        compound_list = self.extract_compounds_list(compounds_dict)
        return compound_list

    def load_cache(self, dir_path):
        self.query_to_atoms = json.load(open(os.path.join(dir_path, f'query2atom_dump.json'), 'r'))
        for key in self.query_to_atoms:
            self.query_to_atoms[key] = set(self.query_to_atoms[key])

        self.query_to_compounds = json.load(open(os.path.join(dir_path, f'query2compound_dump.json'), 'r'))
        print(f'Loaded {len(self.query_to_atoms)} from {dir_path}')

    def dump_cache(self, dir_path):
        for key in self.query_to_atoms:
            self.query_to_atoms[key] = list(self.query_to_atoms[key])
        json.dump(self.query_to_atoms, open(os.path.join(dir_path, f'query2atom_dump.json'), 'w'),
                  ensure_ascii=False, indent=4)

        json.dump(self.query_to_compounds, open(os.path.join(dir_path, f'query2compound_dump.json'), 'w'),
                  ensure_ascii=False, indent=4)
        print(f'Atom and compound dump is saved to {dir_path}')
