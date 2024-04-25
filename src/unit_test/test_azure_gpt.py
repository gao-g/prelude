import os
import json
import requests
import unittest


class TestOAIAzure(unittest.TestCase):
    @unittest.skip("never worked")
    def test_35_notworking(self):
        # deployment_name = "text-davinci-003"
        deployment_name = "gpt-35-turbo"
        base_url = os.getenv("GCR_GPT_URL")
        api_key = os.getenv("GCR_GPT_KEY")

        print(base_url)
        url = base_url + "/openai/deployments/" + deployment_name + "/completions?api-version=2022-12-01"


        prompt = "Abraham Lincoln was"
        payload = {"prompt": prompt,
                "temperature": 0.0,
                "max_token": 20,
                }

        r = requests.post(url,
                        headers={"api-key": api_key,
                                "Content-Type": "application/json"},
                        json=payload)
        response = json.loads(r.text)

        # TODO: not working
        self.assertEqual(response['error']['code'], 'OperationNotSupported')

    @unittest.skip("old openai version < 1.0")
    def test_35_working(self):
        import openai
        self.assertEqual(openai.__version__, '0.28.0')
        openai.api_type = "azure"
        openai.api_version = "2023-05-15" 
        openai.api_base = os.getenv("GCR_GPT_URL")  # Your Azure OpenAI resource's endpoint value.
        openai.api_key = os.getenv("GCR_GPT_KEY")

        response = openai.ChatCompletion.create(
            engine="gpt-35-turbo", # The deployment name you chose when you deployed the GPT-3.5-Turbo or GPT-4 model.
            messages=[
                {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                {"role": "user", "content": "Who were the founders of Microsoft?"}
            ]
        )

        self.assertFalse('error' in response)
        self.assertEqual(response['model'], 'gpt-35-turbo')
        self.assertEqual(response['object'], 'chat.completion')
        self.assertIsNotNone(response['choices'][0]['message']['content'])
        self.assertGreater(len(response['choices'][0]['message']['content']), 0)
        print(response['choices'][0]['message']['content'])

    @unittest.skip("new openai version > 1.0")
    def test_35_working_post1(self):
        import openai
        self.assertEqual(openai.__version__[0], '1')

        client = openai.AzureOpenAI(
            api_key = os.getenv("GCR_GPT_KEY"),
            api_version = "2023-05-15",
            azure_endpoint = os.getenv("GCR_GPT_URL")
        )

        # response = client.chat.completions.create(
        #     model="gpt-35-turbo", # model = "deployment_name".
        #     messages=[
        #         {"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
        #         {"role": "user", "content": "Who were the founders of Microsoft?"}
        #     ],
        # )
        response = client.completions.create(
            model="gpt-35-turbo-instruct",
            prompt="Who were the founders of Microsoft?",
            logprobs=1,
        )

        self.assertFalse('error' in response)
        self.assertIsNotNone(response.choices[0].text)
        self.assertGreater(len(response.choices[0].text), 0)
        self.assertGreater(len(response.choices[0].logprobs.text_offset),0)
        print(response.choices[0].text)

    @unittest.skip("using wrapper - works")
    def test_wrapper35(self):
        from src.language_models.azure_gpt import AzureGPT
        llm_wrapper:AzureGPT = AzureGPT()
        resp = llm_wrapper.get_response_given_prompt("who were the founders of microsoft?")
        self.assertIn("Gates", resp)
        self.assertIn("Allen", resp)

    @unittest.skip("using wrapper - works")
    def test_wrapper4(self):
        from src.language_models.azure_gpt import AzureGPT
        llm_wrapper:AzureGPT = AzureGPT('gpt-4')
        resp = llm_wrapper.get_response_given_prompt("who were the founders of microsoft?")
        self.assertIn("Gates", resp)
        self.assertIn("Allen", resp)

    @unittest.skip("using wrapper - works")
    def test_wrapper_logprob(self):
        from src.language_models.azure_gpt import AzureGPT
        llm_wrapper:AzureGPT = AzureGPT()
        resp = llm_wrapper.get_logprob("who were the founders of microsoft?", "", 1)
        self.assertIsNotNone(resp)
        self.assertTrue(type(resp), float)

    @unittest.skip("using wrapper - works")
    def test_wrapper_length(self):
        from src.language_models.azure_gpt import AzureGPT
        llm_wrapper:AzureGPT = AzureGPT()
        resp = llm_wrapper.get_prompt_length("who were the founders of microsoft?")
        self.assertIsNotNone(resp)
        self.assertTrue(resp, 7)


if __name__ == '__main__':
    unittest.main()