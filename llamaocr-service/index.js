import { config } from 'dotenv';
import { ocr } from 'llama-ocr';

config();

const imagePath = process.argv[2];
if (!imagePath) {
  console.error('No image file path provided.');
  process.exit(1);
}

async function extractText() {
  try {
    const markdown = await ocr({
      filePath: imagePath,
      apiKey: process.env.TOGATHER_API_KEY,
    });
    console.log(markdown);
  } catch (error) {
    console.error('OCR failed:', error);
    process.exit(1);
  }
}

extractText();
