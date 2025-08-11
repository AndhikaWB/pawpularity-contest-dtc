import io
from PIL import Image
import streamlit as st

import requests
from _pydantic.serve import ServeRequest, ServeResponse


features = [
    # Get all features except the image
    ServeRequest.model_fields[key].alias
    for key in ServeRequest.model_fields
    if ServeRequest.model_fields[key].alias != 'Image'
]

features_desc = [
    # Get all feature descriptions
    ServeRequest.model_fields[key].description
    for key in ServeRequest.model_fields
    if ServeRequest.model_fields[key].alias != 'Image'
]

st.subheader('Upload an Image', divider = 'rainbow')

img_uploaded = st.file_uploader(
    'Choose the best/cutest image you have of your pet.',
    type = ['jpg', 'jpeg', 'png']
)

st.subheader('Features Selection', divider = 'rainbow')

if img_uploaded:
    img_uploaded = Image.open(img_uploaded)
    img_uploaded = img_uploaded.convert('RGB')

    img_bytes = io.BytesIO()
    img_uploaded.save(img_bytes, 'JPEG')

    st.image(img_bytes)

features_selected = st.pills(
    'Select zero or more features that truly describe the image:',
    options = [ i for i in range(len(features_desc)) ],
    selection_mode = 'multi',
    format_func = lambda opt: features[opt]
)

if features_selected:
    st.warning(
        'I confirm that these statements below are true:\n\n' + 
        ''.join([ f'- {features_desc[i]}\n' for i in features_selected ])
    )
else:
    st.error(
        'No feature is selected yet.\n\n **Why is this needed?** These features '
        'will be send to the model along with the image. Not selecting any feature '
        'or faking the features will result in inaccurate feedback/prediction.'
    )

predict = st.button(
    'Predict' if features_selected else 'Predict Anyway',
    type = 'primary' if features_selected else 'secondary',
    disabled = False if img_uploaded else True
)

if predict and img_uploaded:
    with st.spinner('Retrieving Prediction...', show_time = True):
        response = requests.post(
            'http://127.0.0.1:8765/predict',
            # This must match the schema provided in the backend (FastAPI)
            data = {features[i]: True for i in features_selected},
            files = {'Image': img_bytes.getvalue()}
        )

    with st.container(border = True):
        st.subheader('Result', divider = 'rainbow')

        if response.status_code == 200:
            pred_result = ServeResponse(**response.json())
            pred_result = round(pred_result.result)
            pred_text = f'{pred_result} out of 100'

            if pred_result < 60:
                pred_text = f'**:red[{pred_text}]**'
                st.markdown(
                    'Aww... Too bad, but your predicted pet pawpularity score is only '
                    f'{pred_text}. For the sake of your pet, maybe you should take '
                    'another photo?'
                )
            elif pred_result <= 80:
                pred_text = f'**:orange[{pred_text}]**'
                st.markdown(
                    f'Not bad... Your predicted pet pawpularity score is {pred_text}, '
                    'but some things certainly can be improved. Perhaps you may want '
                    'to take another photo?'
                )
            else:
                pred_text = f'**:green[{pred_text}]**'
                st.markdown(
                    f'Fantastic! Your predicted pet pawpularity score is {pred_text}. '
                    ' Are you sure you don\'t want to adopt it by yourself? Your pet '
                    'might become a superstar later...'
                )
        else:
            st.markdown(
                'Oops! Something has gone wrong... Expected response status code 200, '
                f'but received status code {response.status_code} ({response.reason}) '
                'instead.'
            )

        with st.expander('See raw response'):
            st.write(response.json())