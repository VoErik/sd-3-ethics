# sd-3-ethics

This project uses **Stable Diffusion 3.5** to generate images of people across different **ethnicities**, **genders**, and **professions** in order to **analyze potential biases** in the model's outputs. The goal is to assess how demographic descriptors affect the quality, style, and stereotypical portrayals in the generated images.

## 📌 Project Objectives

- Generate images based on combinations of prompts involving identity and profession (e.g., _“Male hispanic engineer”_).
- Analyze and document any visual biases, such as stereotypical representations or underrepresentation.
- Evaluate the fairness and generalization of diffusion-based generative models in portraying social categories.

## 🧠 Background

Generative models like Stable Diffusion are trained on large-scale internet data, which may contain embedded social and cultural biases. This project seeks to make those biases visible by systematically prompting the model and analyzing its outputs.

## Settings

* Defined in the `SDConfig`. Generated images and prompts are saved in the `./images` directory by default. 
* We use the `stabilityai/stable-diffusion-3.5-large-turbo` model by default. Be aware that you need to get access to the models first (https://huggingface.co/stabilityai)

