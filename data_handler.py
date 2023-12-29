import json
import torch

class EmbaseNLI(torch.utils.data.Dataset):
    def __init__(self) -> None:

        self.data = self.read_and_filter('./els_data/NER/Embase_NER/Embase_NER_test.json')
        self.data = [{'instruction': entry['instruction'], 'premise': entry['input']['premise'].encode('utf-8').decode('unicode-escape'), 'hypothesis': entry['input']['hypothesis'], 'output': entry['output']} for entry in self.data]
        for entry in self.data:
            if entry['output'] == 'TRUE':
                entry['output'] = 'entailment'
            elif entry['output'] == 'FALSE':
                entry['output'] = 'contradiction'
            else:
                entry['output'] = 'neutral'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not (
            self.has_special_characters(item['input']['premise']) or
            self.has_special_characters(item['input']['hypothesis'])
        )]
        return filtered_list

class EmbaseNER(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./Embase_NER_samples.json')
        self.data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output'], 'entity_type': entry['entity_type'], 'structured_output': entry['structured_output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not self.has_special_characters(item['input'])]
        return filtered_list
    
class EmbaseOQA(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/OpenBookQA/Embase_OpenBookQA/Embase_OpenBookQA_test.json')
        self.data = [{'instruction': entry['instruction'], 'context': entry['input']['context'], 'question': entry['input']['question'], 'output': entry['output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

        filtered_list = [item for item in data if not (
            self.has_special_characters(item['input']['context']) or
            self.has_special_characters(item['input']['question'])
        )]

        return filtered_list

class EmbaseRE(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/RE/Embase_RE/RE_Embase_samples.json')
        self.data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not self.has_special_characters(item['input'])]
        return filtered_list



class ReaxysCQA(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/ClosedBookQA/Reaxys_ClosedBookQA_test.json')
        self.data = [{'question': entry['input'], 'answer': entry['output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not self.has_special_characters(item['input'])]
        return filtered_list

class ReaxysAR(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/AnaphoraResolution/Reaxys_AnaphoraResolution_test.json')
        self.data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output']} for entry in self.data]    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not self.has_special_characters(item['input'])]
        return filtered_list

class ReaxysNER(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/NER/Reaxys_NER/Reaxys_NER_test.json')
        self.data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output']} for entry in self.data]    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if (not self.has_special_characters(item['input']) or not self.has_special_characters(item['output']))]
        return filtered_list



class ResNetCQA(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/ClosedBookQA/ResNet_ClosedBookQA_all_test.json')
        self.data = [{'input': entry['input'], 'output': entry['output']} for entry in self.data]    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if (not self.has_special_characters(item['input']) or not self.has_special_characters(item['output']))]
        return filtered_list

class ResNetNER(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/NER/ResNet_NER/ResNet_NER_test.json')
        self.data = [{'input': entry['input'], 'output': entry['output'], 'structured_output': entry['NER_output']} for entry in self.data]    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if (not self.has_special_characters(item['input']) or not self.has_special_characters(item['output']))]
        return filtered_list

class ResNetOQA(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./els_data/OpenBookQA/ResNet_OpenBookQA/ResNet_OpenBookQA_test.json')
        self.data = [{'question': entry['input']['question'], 'context': entry['input']['context'], 'output': entry['output']} for entry in self.data]    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if (not self.has_special_characters(item['input']['question']) or not self.has_special_characters(item['output']))]
        return filtered_list
    