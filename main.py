import io
from typing import List
from bson import ObjectId
import pdf2image
import pymongo
from pymongo import MongoClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
import uvicorn
import os
from pdf2image import convert_from_path
import boto3
from botocore.exceptions import ClientError
import secrets
import logging
import base64
from openai import OpenAI
import boto3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# OpenAI API Key
api_key = os.environ.get("OPENAI_API_KEY")

AWS_BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")
AWS_REGION = os.environ.get("AWS_REGION")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")

uri = os.environ.get("MONGO_URI")
client = MongoClient(uri)

DB_NAME = "donotfail"

try:
    database = client.get_database(DB_NAME)
    slides = database.get_collection("slides")
    slideImages = database.get_collection("slide_images")

except Exception as e:
    raise Exception(
        "Unable to connect to MongoDB due to the following error: ", e)

client = OpenAI(api_key=api_key)


class Slide(BaseModel):
    _id: str = Field(default_factory=ObjectId)
    name: str = Field(...)
    pdf_url: str = Field(None, allow_none=True)

# Linked list??


class SlideImages(BaseModel):
    _id: str = Field(default_factory=ObjectId)
    slide_id: str = Field(...)
    image_url: str = Field(...)
    order: int = Field(...)
    generated_text: str = Field(None, allow_none=True)
    audio_url: str = Field(None, allow_none=True)


class AudioRequest(BaseModel):
    slide_image_id: str

# Create


s3_client = boto3.client(
    's3',
    region_name=os.environ['AWS_REGION'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']


)

allowed_file_types = ["application/pdf", "image/jpeg", "image/png"]
max_file_size = 1048576 * 100  # 10 MB


def generate_file_name(bytes_length=32):
    return secrets.token_hex(bytes_length)


@app.post("/slides")
async def create_slide(slide: Slide):
    print("/slides")
    print("slide: ", slide)
    slide_dict = slide.dict()
    result = slides.insert_one(slide_dict)
    slide_dict["_id"] = str(result.inserted_id)
    # return status, message, and the created slide
    return {"status_code": 201, "status": "success", "message": "Slide created successfully", "data": slide_dict}


# Read


@app.get("/slides/{slide_id}")
async def read_slide(slide_id: str):
    slide = slides.find_one({"_id": ObjectId(slide_id)})
    print("slide: ", slide)
    slide["_id"] = str(slide["_id"])
    if slide:
        return {"status": "success", "data": slide, "message": "Slide found", "status_code": 200}
    else:
        return {"status": "fail", "message": "Slide not found", "status_code": 404}


@app.get("/slides/{slide_id}/images")
async def read_slide_images(slide_id: str):
    slide_images_list = []
    for slide_image in slideImages.find({"slide_id": slide_id}):
        print("slide_image: ", slide_image)
        slide_image["_id"] = str(slide_image["_id"])
        slide_images_list.append(slide_image)
    return {"status": "success", "data": slide_images_list, "status_code": 200}

# List


@app.get("/slides")
async def list_slides():
    print("list_slides")
    slides_list = []
    for slide in slides.find():
        print("slide: ", slide)
        slide["_id"] = str(slide["_id"])
        slides_list.append(slide)
    return {"status": "success", "data": slides_list, "status_code": 200}

# Update


@app.put("/slides/{slide_id}")
async def update_slide(slide_id: str, slide: Slide):
    print("slide_id: ", slide_id)
    print("slide: ", slide)
    slide_dict = slide.dict()
    print("slide_dict: ", slide_dict)
    result = slides.find_one_and_update(
        {"_id": ObjectId(slide_id)}, {"$set": slide_dict}, return_document=pymongo.ReturnDocument.AFTER)
    result["_id"] = str(result["_id"])
    if result:
        return {"message": "Slide updated successfully", "status_code": 200, "data": result}
    else:
        return {"message": "Slide not found", "status_code": 404}


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
        print("response: ", response)
        logging.info("response: ", response)
    except ClientError as e:
        logging.error(e)
        print("e: ", e)
        return False
    return True


def process_pdf_to_images(pdf_path, output_dir, dpi=500):
    pages = convert_from_path(pdf_path, dpi)
    print(f'Number of pages: {len(pages)}')

    for count, page in enumerate(pages):
        print(f'Processing page {count}')
        page.save(f'{output_dir}/out-{count}.png', 'PNG')


@app.post("/slides/{slide_id}/pdf-to-images")
async def convert_pdf_to_images(slide_id: str):
    slide = slides.find_one({"_id": ObjectId(slide_id)})
    print("slide: ", slide)
    if not slide:
        return {"message": "Slide not found", "status_code": 404}
    pdf_url = slide.get("pdf_url")
    print("pdf_url: ", pdf_url)
    if not pdf_url:
        return {"message": "PDF URL not found", "status_code": 404}
    # if folder not created, create it
    # if not os.path.exists("./temp"):
    #     print("Creating temp folder")
    #     os.makedirs("./temp")
    # if not os.path.exists(f"./temp/{slide_id}"):
    #     print
    #     os.makedirs(f"./temp/{slide_id}")

    # print("Downloading PDF")
    # # download pdf and convert to images
    # pdf_file_name = f"./temp/{slide_id}/slide.pdf"
    # response = requests.get(pdf_url)
    # print("response: ", response)
    # with open(pdf_file_name, 'wb') as file:
    #     file.write(response.content)

    # print("Converting PDF to images")
    # process_pdf_to_images(pdf_file_name, f"./temp/{slide_id}")
    # print("PDF converted to images")
    # # upload images to S3
    # images = os.listdir(f"./temp/{slide_id}")

    response = requests.get(pdf_url, timeout=30)
    images = pdf2image.convert_from_bytes(response.content, fmt="png")

    print("images1: ", images)
    index = 0
    # filter out non-image files
    # images = [image for image in images if image.endswith(".png")]
    # print("images2: ", images)
    # sorted_images = sorted(images, key=lambda x: int(
    #     x.split("-")[1].split(".")[0]))
    for image in images:
        try:
            print("Uploading image: ", image)
            # image_path = f"./temp/{slide_id}/{image}"
            file_name = generate_file_name()
            aws_path = f"slides/{slide_id}/images/{file_name}.png"

            in_mem_file = io.BytesIO()
            # Define the missing variable
            # key = "your_key_here"
            image.save(in_mem_file, format="PNG")

            in_mem_file.seek(0)

            client_s3 = boto3.client('s3')
            res = client_s3.upload_fileobj(
                in_mem_file,
                AWS_BUCKET_NAME,
                aws_path,
            )

            print("res: ", res)
            # if res.get("ResponseMetadata").get("HTTPStatusCode") == 200:
            url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{aws_path}"
            print("url: ", url)
            slide_images = SlideImages(
                slide_id=slide_id, image_url=url, order=index)
            index += 1
            print("slide_images: ", slide_images)
            slide_images_dict = slide_images.dict()
            result = slideImages.insert_one(slide_images_dict)
            slide_images_dict["_id"] = str(result.inserted_id)
            print("slide_images_dict: ", slide_images_dict)
            # delete the image from the local folder
            # os.remove(image_path)
        except Exception as e:
            print("Error uploading image: ", e)
            # Handle the exception here

    # os.remove(pdf_file_name)
    # os.rmdir(f"./temp/{slide_id}")

    return {"message": "PDF converted to images", "status_code": 200}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def process_image(image_url):
    PROMPT = """
    Describe and explain this lecture slide, no fluff, buzzwords or jargon.
    """
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ],
            }
        ],
        max_tokens=500,
    )

    return response


@app.post("/generate-image-text/{slide_image_id}")
async def serve_text(slide_image_id: str):
    print("*** /generate-image-text ***")
    slide_image = slideImages.find_one({"_id": ObjectId(slide_image_id)})
    print("slide_image: ", slide_image)
    image_url = slide_image.get("image_url")
    print("image_url: ", image_url)
    if not image_url:
        return {"error": "Image not found"}

    if slide_image.get("generated_text"):
        return {"status": "success", "data": slide_image.get("generated_text"), "status_code": 200}
    print("Processing image")
    try:
        response = process_image(image_url)
        print("response raw: ", response)
        generated_text = response.choices[0].message.content
        print("generated_text: ", generated_text)
        result = slideImages.find_one_and_update(
            {"_id": ObjectId(slide_image_id)}, {"$set": {"generated_text": generated_text}})
        print("result: ", result)
        return {"status": "success", "data": generated_text, "status_code": 200}
    except Exception as e:
        return {"error": str(e)}


class SeachRequest(BaseModel):
    # context: List[int]
    context: str
    question: str


def answer_question(context, question):
    PROMPT = """
    You are a helpful assistant that can answer questions. If you don't know the answer, you can say 'I don't know'. Or if you don't have all the information, just tell me what you can.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system",
                "content": PROMPT
            },
            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": context
            #         }
            #     ]
            # },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=300,
    )
    return response


@app.post("/search")
async def search_question(request: SeachRequest):
    print("*** /search ***")
    context = request.context
    CONTEXT = ""
    # for i in range(len(context)):
    #     fileName = f'out-{context[i]}.jpg'
    #     results = slideImages.find({"image_url": fileName})
    #     if results.count() == 0:
    #         continue
    #     slide = context[i] + 1

    #     CONTEXT += f"SLIDE {slide}: \n" + results[0]["data"] + "\n\n"

    question = request.question
    response = answer_question(CONTEXT, question)
    # print("response: ", response)

    return response


@app.post("/generate-audio")
async def generate_audio(request: AudioRequest):
    print("*** /audio ***")

    slide_image_id = request.slide_image_id
    slide_image = slideImages.find_one({"_id": ObjectId(slide_image_id)})
    # print("slide_image: ", slide_image)
    if not slide_image:
        return {"error": "Slide Image not found"}

    generated_text = slide_image.get("generated_text")
    # print("generated_text: ", generated_text)

    if not generated_text:
        return {"error": "Generated Text not found"}

    if slide_image.get("audio_url"):
        return {"status": "success", "data": slide_image.get("audio_url"), "status_code": 200}

    try:
        print("Generating audio file")
        voice = "aura-asteria-en"  # "aura-helios-en"  # "aura-luna-en" , "aura-asteria-en"
        url = f"https://api.deepgram.com/v1/speak?model={voice}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        data = {
            "text": generated_text
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        audio_blob = response.content

        # upload the audio to S3
        file_name = generate_file_name()
        aws_path = f"slides/{slide_image.get('slide_id')}/audio/{file_name}.mp3"
        s3 = boto3.resource('s3')
        res = s3.Object(AWS_BUCKET_NAME, aws_path).put(
            Body=audio_blob)
        print("res: ", res)
        if res.get("ResponseMetadata").get("HTTPStatusCode") == 200:
            url = f"https://{AWS_BUCKET_NAME}.s3.amazonaws.com/{aws_path}"
            print("url: ", url)
            result = slideImages.find_one_and_update(
                {"_id": ObjectId(slide_image_id)}, {"$set": {"audio_url": url}})
            print("result: ", result)
            return {"status": "success", "data": url, "status_code": 200}
        else:
            return {"error": "Error uploading audio to S3"}

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def delete_aws_file(key):
    try:
        s3 = boto3.resource('s3')
        res = s3.Object(AWS_BUCKET_NAME, key).delete()
        print("res: ", res)
        if res.get("ResponseMetadata").get("HTTPStatusCode") == 204:
            return True
        else:
            return False
    except Exception as e:
        return False


@app.delete("/slides/{slide_id}")
async def delete_slide(slide_id: str):
    print("*** /delete_slide ***")
    slide = slides.find_one({"_id": ObjectId(slide_id)})
    pdf_url = slide.get("pdf_url")

    print("slide: ", slide)
    result = slides.delete_one({"_id": ObjectId(slide_id)})
    print("result: ", result)
    if result.deleted_count == 1:
        if pdf_url:
            pdf_key = pdf_url.split(
                "amazonaws.com/")[1]
            delete_aws_file(pdf_key)
        # delete all images and audio files associated with the slide
        for slide_image in slideImages.find({"slide_id": slide_id}):
            audio_url = slide_image.get("audio_url")
            if audio_url:
                print("audio_url: ", audio_url)
                audio_key = audio_url.split(
                    "amazonaws.com/")[1]
                delete_aws_file(audio_key)
            print("slide_image: ", slide_image)
            image_key = slide_image.get("image_url").split(
                "amazonaws.com/")[1]
            delete_aws_file(image_key)
        slideImages.delete_many({"slide_id": slide_id})
        return {"message": "Slide deleted successfully", "status_code": 200}
    else:
        return {"message": "Slide not found", "status_code": 404}

# ping


@app.get("/ping")
async def ping():
    return {"status": "success", "data": "pong", "status_code": 200}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
