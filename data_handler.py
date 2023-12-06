from abc import ABC, abstractmethod
import json

class AbstractDataFetcher(ABC):
    @abstractmethod
    def fetch_nli(self, file_name: str) -> list:
        pass

    @abstractmethod
    def fetch_ner(self, file_name: str) -> list:
        pass

    @abstractmethod
    def fetch_re(self, file_name: str) -> list:
        pass

    @abstractmethod
    def fetch_open_qa(self, file_name: str) -> list:
        pass

    def handle_unicode_escape(obj):
        if isinstance(obj, str):
            return obj.encode().decode('unicode-escape')
        return obj

    def read_json_file(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        return data


class EmbaseDataFetcher(AbstractDataFetcher):
    def fetch_nli(self, file_name: str) -> list:
        data = self.read_json_file(file_name)
        data = [{'instruction': entry['instruction'], 'premise': entry['input']['premise'].encode('utf-8').decode('unicode-escape'), 'hypothesis': entry['input']['hypothesis'], 'output': entry['output']} for entry in data if entry['task'] == 'natural language inference' and entry['DataAsset'] == 'Embase']
        return data

    def fetch_ner(self, file_name: str) -> list:
        data = self.read_json_file(file_name)
        data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output']} for entry in data if entry['task'] == 'Named Entity Recognition' and entry['DataAsset'] == 'Embase']
        return data

    def fetch_re(self, file_name: str) -> list:
        data = self.read_json_file(file_name)
        data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output']} for entry in data if entry['task'] == 'Relation Extraction' and entry['DataAsset'] == 'Embase']
        return data

    def fetch_open_qa(self, file_name: str) -> list:
        data = self.read_json_file(file_name)
        data = [{'instruction': entry['instruction'], 'context': entry['input']['context'], 'question': entry['input']['question'], 'output': entry['output']} for entry in data if entry['task'] == 'Open Book QA' and entry['DataAsset'] == 'Embase']
        return data




embase_fetcher = EmbaseDataFetcher()
nli_data = embase_fetcher.fetch_nli('./els_data/NaturalLanguageInference/Embase_NLI/Inference_Embase_samples.json')
input_prompt = f'''Question: {nli_data[3]['instruction']}

Premise: {nli_data[3]['premise']}


Hypotehsis: {nli_data[3]['hypothesis']}

Provide your answer as one of the following only: TRUE, FALSE, or UNDETERMINED.
'''
print(input_prompt)

print(nli_data[3]['output'])
