# sd-3-ethics

This project uses **Stable Diffusion 3.5** to generate images of people across different **ethnicities**, **genders**, and **professions** in order to **analyze potential biases** in the model's outputs. The goal is to assess how demographic descriptors affect the quality, style, and stereotypical portrayals in the generated images.

## 📌 Project Objectives

- Generate images based on combinations of prompts involving identity and profession (e.g., _“Male hispanic engineer”_).
- Analyze and document any visual biases, such as stereotypical representations or underrepresentation.
- Evaluate the fairness and generalization of diffusion-based generative models in portraying social categories.

## 🧠 Background

Generative models like Stable Diffusion are trained on large-scale internet data, which may contain embedded social and cultural biases. This project seeks to make those biases visible by systematically prompting the model and analyzing its outputs.

## Settings

* You need a HuggingFace access token. See how you can generate it [here](https://huggingface.co/docs/hub/security-tokens).
* Create an `.env` file and save the token there. If you name it other than `HF_KEY`, you need to modify the code accordingly.
* The configuration for the model is set in `SDConfig`. 
* We use the `stabilityai/stable-diffusion-3.5-large-turbo` model by default. Be aware that you need to get access to the models first (https://huggingface.co/stabilityai)
* Generated images and prompts are saved in the `./images` directory by default. 

## Note
This is an ongoing project. Results as well as the project paper will also be added to this repo.

