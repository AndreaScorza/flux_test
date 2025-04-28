from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype="auto"
)
pipe.to("cuda")  # Or "cpu" if no GPU

prompt = "a futuristic city at sunset, ultra detailed"
image = pipe(prompt).images[0]

image.save("flux_output.png")
image.show()
