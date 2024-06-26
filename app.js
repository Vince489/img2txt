const express = require('express');
const multer = require('multer');
const { HfInference } = require('@huggingface/inference');
const path = require('path');

const app = express();
const port = process.env.PORT || 5000;

// Set up Multer for handling file uploads
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

const HF_TOKEN = "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";

// Set up Hugging Face Inference
const inference = new HfInference(HF_TOKEN);

// Set the view engine to EJS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// serve index.ejs
app.get('/', (req, res) => {
  res.render('index');
});

// Handle file upload and generate image
app.post('/upload', upload.single('text'), async (req, res) => {
  try {
    const textDescription = req.body.text;
    const negative_prompt = req.body.negativePrompt;
    const style = req.body.style || 'default';
    const seed = req.body.seed || Math.floor(Math.random() * (943221 - (-873098) + 1) + (-873098));
    const lora_scale = req.body.loraScale || 5;
    const guidanceScale = req.body.guidanceScale || 5;

    if (!textDescription) {
      return res.status(400).send('Text is required');
    }

    const model = 'stabilityai/stable-diffusion-xl-base-1.0';
    let styler;
    let styleModel
    let negativePromptModel

    switch (style) {
      case 'impressionist':
        styler = 'Impressionist'
        styleModel = 'impressionist influenced';
        negativePromptModel = 'realism'
        break;
      case 'cubist':
        styleModel = 'cubist influenced';
        break;
      case 'anime':
        styleModel = 'anime influenced';
        break;
      case 'pixar':
        styleModel = 'pixar influenced';
        break;
      case 'adult':
        styleModel = 'naked_nude_bare_skin';
        break;
      case 'realism':
        styleModel = 'hd_8k_hi_res';
        break;
      default:
        break;
    }

    // Prepend styleModel to textDescription
    const styledTextDescription = `${styleModel}: ${textDescription}`;

    // Prepend negativeModel to negativePrompt
    const negativePromptDescription = `${negativePromptModel}: ${negative_prompt}`;

    // Set default dimensions and num_inference_steps
    let height = 512;
    let width = 512;
    let numInferenceSteps = 30;

    // Adjust dimensions and num_inference_steps based on aspect ratio
    const aspectRatio = req.body.aspectRatio;
    switch (aspectRatio) {
      case '1:1':
        height = 512;
        width = 512;
        num_inference_steps = 50;
        break;
      case '2:3':
        height = 512;
        width = 768;
        numInferenceSteps = 42;
        break;
      case '3:4':
        height = 768;
        width = 512;
        num_inference_steps = 50;
        break;
      case '4:3':
        height = 768;
        width = 768;
        num_inference_steps = 50;
        break;
      case '5:7':
        height = 768;
        width = 1024;
        num_inference_steps = 50;
        break;
      case '7:5':
        height = 1024;
        width = 768;
        num_inference_steps = 50;
        break;
      case '9:16':
        height = 1024;
        width = 1024;
        num_inference_steps = 64;
        break;
      default:
        break;
    }

    const result = await inference.textToImage({
      model: model,
      inputs: styledTextDescription, // Use styledTextDescription here
      parameters: {
        negative_prompt: negativePromptDescription,
        height: height,
        width: width,
        lora_scale: lora_scale,
        seed: seed,
        guidanceScale: guidanceScale,
      }
    });

    const blobContent = await result.arrayBuffer();

    // Convert the ArrayBuffer to a Data URL
    const dataUrl = `data:${result.type};base64,${Buffer.from(blobContent).toString('base64')}`;

    // Render the result using EJS
    res.render('result', {
      dataUrl: dataUrl,
      styleModel: styler,
      prompt: textDescription, // Use styledTextDescription here
      negativePrompt: negative_prompt,
      loraScale: lora_scale,
      seed: seed,
      guidanceScale: guidanceScale,
      aspectRatio: aspectRatio
    });
  } catch (error) {
    console.error('Error during image generation:', error);
    res.status(500).send('Error during image generation');
  }
});


app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});
