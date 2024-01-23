from fastai.vision.all import load_learner
import gradio as gr


def is_cat(x):
    return x[0].is_upper


learn = load_learner("model.pkl")

categories = ("Dog", "Cat")


def classify_image(img):
    _, _, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ["dog.jpg", "cat.jpg"]

gradio_interface = gr.Interface(
    fn=classify_image, inputs=image, outputs=label, examples=examples
)
gradio_interface.launch(inline=False)
