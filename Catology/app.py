from operator import is_not

from accelerate.commands.config.config import description
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from transformers import pipeline
import torch
import re
import numpy as np
from flask import jsonify
from model import RN
import pandas as pd
print(torch.cuda.is_available())

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


class UserInput(FlaskForm):
    input = StringField("Enter description")
    submit = SubmitField("Submit")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict_nn(attributes):

    print("fac predictie")
    rn = RN()
    rn.prepare(attributes)
    rn.train_dynamic(epochs=50)
    rn.save_model()

    weights_input_hidden = np.load('Catology/weights_input_hidden.npy')
    weights_hidden_output = np.load('Catology/weights_hidden_output.npy')
    b1 = np.load('Catology/b1.npy')
    b2 = np.load('Catology/b2.npy')


    input_data = np.array([[
        attributes["Number"],
        attributes["Ext"],
        attributes["Shy"],
        attributes["Calm"],
        attributes["Scared"],
        attributes["Vigilant"],
        attributes["Affectionate"],
        attributes["Friendly"],
        attributes["Solitary"],
        attributes["Aggressive"],
        attributes["PredatorMammal"],
        attributes["Coat"],
        attributes["Intelligence_Score"]
    ]])

    print(f"input data: {input_data}")
    input_data2 = input_data[input_data != -1]
    print(f"input data2: {input_data2}")

    hidden_layer, output_layer = rn.forward(input_data2)
    predicted_class = np.argmax(output_layer, axis=1)[0]
    return predicted_class


def validate_output(generated_text):

    pattern = r"-([\w_]+):\s*(-?\d)"
    matches = re.findall(pattern, generated_text)
    if len(matches) == 13:
        return {attr: int(score) for attr, score in matches}
    else:
        print("Validation failed for output:", generated_text)
        return None


def evaluate_attributes(input_text):


    prompt = f"""
    Evaluate the following characteristics of the animal and assign a score each attribute. 
    Respond ONLY with the scores in the following format and only if you find details about that attribute in text, where you don t find details about attributes put -1.
    The attributes to evaluate are:

    -Number: [between 1 and 5]
    -Ext: [between 0 and 4]
    -Shy: [between 1 and 5]
    -Calm: [between 1 and 5]
    -Scared: [between 1 and 5]
    -Vigilant: [between 1 and 5]
    -Affectionate: [between 1 and 5]
    -Friendly: [between 1 and 5]
    -Solitary: [between 1 and 5]
    -Aggressive: [between 1 and 5]
    -PredatorMammal: [between 0 and 4]
    -Coat: [between 0 and 4]
    -Intelligence_Score: [between 0 and 4]

    where coat refers to the length of the hair
    and number refers to how many cats are in the same household
    Answear with a number in the interval for each attribute you can find in the text, for those that are not sure, you put nothing not event the attribute name. If there is no mention of an attribute, assign a score of 0. No description, 
    Text: "{input_text}"

        """

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    attributes_dict = validate_output(generated_text)

    print(f"Parsed attributes: {attributes_dict}")
    return attributes_dict


def return_cat_description(predicted_class):
    breed_dict = {1: "Bengal",
                  2: "Birman",
                  3: "British Shorthair",
                  4: "Chartreux",
                  5: "European Shorthair",
                  6: "Maine Coon",
                  7: "Persian",
                  8: "Ragdoll",
                  9: "Sphynx",
                  10: "Oriental Shorthair",
                  11: "Turkish Van",
                  }

    return breed_dict[predicted_class]

def return_human_description(predicted_class, human_description):
    prompt = f"""
           You will be given a number and a description of a human. Based on this number i want you to return the full name of the breed.
           The breed will be one of these based on the number:
       "Bengal": 1,
       "Birman": 2,
       "British Shorthair": 3,
       "Chartreux": 4,
       "European Shorthair": 5,
       "Maine Coon": 6,
       "Persian": 7,
       "Ragdoll": 8,
       "Sphynx": 9,
       "Oriental Shorthair": 10,
       "Turkish Van": 11,
       "Other": 12,
       "Unknown": 13 
       Based on the description of the human, i want you to say why this breed would be a good match for this person
       Here is the number: {predicted_class}
       Here is the description of the human: {human_description}
       format the text a bit so it's not just a block of text and don't ask the user anything, just give the breed and the reasoning!!!
               """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text

def return_human_description_ro(predicted_class, human_description):
    prompt = f"""
    O sa ti fie dat un un numar si o descriere a unui om. Bazat pe acest numar vreau sa returnezi numele intreg al acestei rase.
    Rasa de pisici va fi una din acestea bazat pe numarul primit:
       "Bengal": 1,
       "Birman": 2,
       "British Shorthair": 3,
       "Chartreux": 4,
       "European Shorthair": 5,
       "Maine Coon": 6,
       "Persian": 7,
       "Ragdoll": 8,
       "Sphynx": 9,
       "Oriental Shorthair": 10,
       "Turkish Van": 11,
       "Other": 12,
       "Unknown": 13 
       Bazat pe descrierea omului, vreau sa mi zici de ce rasa de pisica numarul {predicted_class} ar avea o compatibilitate foarte buna pentru aceasta persoana.
       Aceasta este descrierea omului {human_description}
       Formateaza textul un pic sa nu fie doar un bloc de text si nu pune nicio intrebare inapoi, doar da rasa si motivul. Raspunde in romana!
               """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text


def get_avg_att(breed_id):
    file_path = 'Catology/output.xlsx'
    data = pd.read_excel(file_path)

    second_row = data.iloc[int(breed_id) - 1]

    breed_dict = {  1:"Bengal",
                    2:"Birman",
                    3:"British Shorthair",
                    4:"Chartreux",
                    5:"European Shorthair",
                    6:"Maine Coon",
                    7:"Persian",
                    8:"Ragdoll",
                    9:"Sphynx",
                    10:"Oriental Shorthair",
                    11: "Turkish Van",
                    }
    att_dict = {}

    columns = data.columns

    for col, value in second_row.items():
        att_dict[col] = value

    breed_name = breed_dict[int(breed_id)]
    att_dict["breed"] = breed_name

    print(f"att_dict: {att_dict}")
    return att_dict

def get_breed_description(attributes_dict):
    prompt = f"""
            You will be given a cat breed and it's attributes. Each attribute will have a score where 0 means the least and 5 the most.
            Based on this, i want you to return the cat breed description considering almost every attribute.
        Only say the description, nothing else!!
        Here is the information: {attributes_dict}
                """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text

def get_breed_description_ro(attributes_dict):
    prompt = f"""
            You will be given a cat breed and it's attributes. Each attribute will have a score where 0 means the least and 5 the most.
            Based on this, i want you to return the cat breed description considering almost every attribute.
        Only say the description, nothing else!!
        Here is the information: {attributes_dict}
        
        Return the text and only the text translated to romanian.

                """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text


def get_breed_comparison(breed1, breed2):
    prompt = f"""
            You will be given two cat breeds with their attributes. Each attribute will have a score where 0 means the least and 5 the most.
            Based on this information I want you to give me a comparison between the two breeds, considering almost every attribute.
            I want a detailed comparison, like a story.
            Only say the comparison, nothing else!!
            Here is the first breed: {breed1}
            Here is the second breed: {breed2}
                    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text

def get_breed_comparison_ro(breed1, breed2):
    prompt = f"""
            You will be given two cat breeds with their attributes. Each attribute will have a score where 0 means the least and 5 the most.
            Based on this information I want you to give me a comparison between the two breeds, considering almost every attribute.
            I want a detailed comparison, like a story.
            Only say the comparison, nothing else!!
            Here is the first breed: {breed1}
            Here is the second breed: {breed2}
            
            Return the text and only the text translated to romanian.
                    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = pipe(messages, max_new_tokens=256)
    generated_text = response[0]["generated_text"][-1]["content"].strip()

    print(f"Generated raw output: {generated_text}")

    return generated_text


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


@app.route('/identify', methods=["GET", "POST"])
def identify():
    if request.method == "POST":
        try:
            data = request.get_json()
            user_input = data.get('input', '').strip()
            lang = data.get('lg')

            print(f"User input: {user_input}")

            attributes = None
            prediction = None
            description = None

            if user_input:
                attributes = evaluate_attributes(user_input)
                if attributes:
                    prediction = predict_nn(attributes)
                    print("prediction: ")
                    print(prediction)
                if prediction:
                    description = return_cat_description(prediction)


                print("here")
                return jsonify({'description': description or "No description available"})
            else:
                return jsonify({'error': "No input provided"}), 400

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500

    form = UserInput()
    return render_template('identify.html', user_input=form)


@app.route('/match', methods=["GET", "POST"])
def match():
    if request.method == "POST":
        try:
            data = request.get_json()
            user_input = data.get('input', '').strip()
            lang = data.get('lg')

            print(f"User input: {user_input}")

            attributes = None
            prediction = None
            description = None

            if user_input:
                attributes = evaluate_attributes(user_input)
                if attributes:
                    prediction = predict_nn(attributes)
                    print("prediction: ")
                    print(prediction)
                if prediction:
                    if lang == 'en':
                        description = return_human_description(prediction, user_input)
                    else:
                        description = return_human_description_ro(prediction, user_input)

                print("here")
                return jsonify({'description': description or "No description available"})
            else:
                return jsonify({'error': "No input provided"}), 400

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500

    return render_template('match.html', user_description=UserInput())


@app.route('/compare', methods=["GET", "POST"])
def compare():
    if request.method == "POST":
        try:
            data = request.get_json()
            description = ""

            if not data:
                return jsonify({'error': "No JSON data received"}), 400

            form_type = data.get('type')
            lang = data.get('lg')

            if form_type == 0:
                breed = data.get('input')
                print(f"breed: {breed}")
                attributes = get_avg_att(breed)

                if lang == 'en':
                    description = get_breed_description(attributes)
                else:
                    description = get_breed_description_ro(attributes)
                print(f"description: {description}")
                return jsonify({'description': description or "No description available"})

            elif form_type == 1:
                breed = data.get('input')
                breed2 = data.get('input2')

                print(breed)
                print(breed2)
                cat1 = get_avg_att(breed)
                cat2 = get_avg_att(breed2)

                if lang == 'en':
                    description = get_breed_comparison(cat1, cat2)
                else:
                    description = get_breed_comparison_ro(cat1, cat2)
                return jsonify({'description': description or "No description available"})
        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({'error': str(e)}), 500
    return render_template('compare.html')



@app.route('/about', methods=["GET"])
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(_, _, debug=True)