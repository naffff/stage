import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import json

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("nielsr/layoutlmv2-finetuned-funsd")

# load image example
dataset = load_dataset("nielsr/funsd", split="test")
labels = dataset.features['ner_tags'].feature.names
id2label = {v: k for v, k in enumerate(labels)}
label2color = {'question': 'blue', 'answer': 'green', 'header': 'orange', 'other': 'violet'}

def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')

    # forward pass
    outputs = model(**encoding)

    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()

    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:, 0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Initialize variables to store questions and answers
    current_question = ""
    current_answer = ""
    questions = []
    answers = []

    for idx, prediction, box in zip(range(len(true_predictions)), true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()

        # If it's a question token, store the question
        if predicted_label == 'question':
            current_question = processor.decode(encoding.input_ids[0][idx: idx + 1])  # Handling subword tokens
        # If it's an answer token, store the answer and draw the previous question with its answer
        elif predicted_label == 'answer':
            current_answer = processor.decode(encoding.input_ids[0][idx: idx + 1])  # Handling subword tokens

            # Draw the previous question and answer
            if current_question != "":
                draw.rectangle(box, outline=label2color['question'])
                draw.text((box[0] + 10, box[1] - 10), text=current_question, fill=label2color['question'], font=font)
                draw.text((box[0] + 10, box[1] + 10), text=current_answer, fill=label2color['answer'], font=font)

                # Add the question and answer to their respective lists
                questions.append(current_question)
                answers.append(current_answer)

            # Reset variables for the next question-answer pair
            current_question = ""
            current_answer = ""

    # Save the extracted questions and answers to a JSON file
    output_data = {
        "questions": questions,
        "answers": answers
    }

    # Provide the filename where you want to save the JSON data
    output_filename = "output_data.json"

    # Save the JSON data to the file
    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file)

    return image

if __name__ == "__main__":
    # Example usage:
    image_path = "invoice.png"  # Replace with the actual image path
    annotated_image = process_image(image_path)
    annotated_image.save("annotated_image.jpg")
