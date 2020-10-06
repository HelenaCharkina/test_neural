
from tensorflow import keras
from GetString import get_string


# model = create_cnn()
# train_model(model)
# model.save('model/model.h5')

model = keras.models.load_model('model/model.h5')
result = get_string(model, "source/test2.jpg")
print(result)
