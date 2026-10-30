import unittest

from app.models.unlimited_ocr_2 import UnlimitedOCR2


class TestUnlimitedOCR2(unittest.TestCase):
    def setUp(self):
        self.model = UnlimitedOCR2(
            {"model_id": "unlimited_ocr_2", "display_name": "Unlimited-OCR-2"}
        )
        self.model.backend = "transformers"

    def test_document_coordinates_and_cross_page_groups(self):
        result = self.model._parse_document(
            "<PAGE><BLOCK>title<LEVEL1>[100,200,900,400]<FORK>Heading"
            "<BLOCK>cross_text[0,400,1000,900]<FORK>First half "
            "<PAGE><BLOCK>append_text[0,0,1000,400]<FORK>second half.",
            [(200, 100), (400, 300)],
        )
        first, second = [
            p['prunedResult']['parsing_res_list']
            for p in result['layoutParsingResults']
        ]
        self.assertEqual(first[0]['block_label'], 'doc_title')
        self.assertEqual(first[0]['block_bbox'], [20, 20, 180, 40])
        self.assertEqual(second[0]['block_bbox'], [0, 0, 400, 120])
        self.assertEqual(
            first[1]['global_group_id'], second[0]['global_group_id']
        )
        self.assertNotEqual(
            first[1]['global_block_id'], second[0]['global_block_id']
        )
        self.assertIn('First half', result['markdown']['text'])
        self.assertIn('second half.', result['markdown']['text'])

    def test_table_merge_and_blank_page(self):
        result = self.model._parse_document(
            '<PAGE><BLOCK>cross_table[0,0,1000,1000]<FORK>'
            '<table><tr><td>Header</td></tr><tr><td>A</td></tr></table>'
            '<PAGE><BLOCK>append_table[0,0,1000,1000]<FORK>'
            '<table><tr><td>Header</td></tr><tr><td>B</td></tr></table>'
            '<PAGE>',
            [(200, 100)] * 3,
        )
        self.assertEqual(len(result['layoutParsingResults']), 3)
        self.assertEqual(
            result['layoutParsingResults'][2]['prunedResult'][
                'parsing_res_list'
            ],
            [],
        )
        self.assertEqual(result['markdown']['text'].count('Header'), 1)
        self.assertIn('<td>B</td>', result['markdown']['text'])

    def test_invalid_boxes_and_missing_pages(self):
        result = self.model._parse_document(
            '<PAGE><BLOCK>text[-10,0,1100,1000]<FORK>Clipped'
            '<BLOCK>text[800,100,100,900]<FORK>Reversed',
            [(200, 100)],
        )
        blocks = result['layoutParsingResults'][0]['prunedResult'][
            'parsing_res_list'
        ]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['block_bbox'], [0, 0, 200, 100])
        with self.assertRaisesRegex(ValueError, 'page count'):
            self.model._parse_document('<PAGE>One page', [(200, 100)] * 2)

    def test_upstream_trailing_empty_page_marker(self):
        result = self.model._parse_document(
            '<PAGE><BLOCK>text[0,0,1000,1000]<FORK>Page one<PAGE>',
            [(200, 100)],
        )
        self.assertEqual(len(result['layoutParsingResults']), 1)
        self.assertEqual(
            result['layoutParsingResults'][0]['markdown']['text'], 'Page one'
        )
        with self.assertRaisesRegex(ValueError, 'page count'):
            self.model._parse_document(
                '<PAGE>Page one<PAGE>Unexpected content', [(200, 100)]
            )
