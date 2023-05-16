import tensorflow_hub as hub
import tensorflow as tf

def serving(input):
    hub_url = "https://tfhub.dev/google/bird-vocalization-classifier/2"
    model = hub.load(hub_url)
    
    results = model.tf_infer(input)
    return results