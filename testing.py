import unittest
from nltk.corpus import stopwords

# import your functions from app.py
from app import sentence_similarity, read_article


class TestTextSummarizer(unittest.TestCase):

    def setUp(self):
        # load stopwords once for all tests
        self.stop_words = stopwords.words("english")

    # ---------------- TEST 1 ---------------- #
    def test_sentence_similarity(self):
        sent1 = ["flask", "is", "fast"]
        sent2 = ["flask", "is", "simple"]

        similarity = sentence_similarity(sent1, sent2, self.stop_words)

        self.assertTrue(0 <= similarity <= 1)

    # ---------------- TEST 2 ---------------- #
    def test_read_article(self):
        text = "Flask is fast. It is simple."
        result = read_article(text)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    # ---------------- TEST 3 ---------------- #
    def test_empty_input(self):
        text = ""
        result = read_article(text)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()