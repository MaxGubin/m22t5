import dataset
import unittest

from transformers import T5Tokenizer

class TestDatasetCreation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    def test_batch_prepare(self):
        input_samples = [
            {"source": "I is right", "corrected": "I am wrong" },
            {"source": "I is right", "corrected": "I am wrong" },
            {"source": "I is right", "corrected": "I am wrong" },
        ]
        prepared = dataset.convert_to_features(self.tokenizer, input_samples)
        self.assertIn("input_ids", prepared)
        self.assertIn("target_ids", prepared)

        self.assertEqual((3,512), prepared["input_ids"].shape)

    def test_batch_generate(self):
        i = 0
        for btch in dataset.create_dataset(self.tokenizer, "converted.test.json", 16):
            self.assertIn("attention_mask", btch)
            self.assertEqual((16,512), btch["target_attention_mask"].shape)
            i += 1
            if i == 100:
                break

if __name__ == '__main__':
    print("In test")
    unittest.main()
