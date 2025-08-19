import io
import dotenv
from PIL import Image
from pathlib import Path

import requests
import streamlit as st
from pawpaw.pydantic_.serve import ServeConf, ServeRequest, ServeResponse


def image_section(features: dict[str, str]):
    st.subheader('Upload an Image', divider = 'rainbow')

    img_uploaded = st.file_uploader(
        'Choose the best/cutest image you have of your pet.',
        type = ['jpg', 'jpeg', 'png']
    )

    st.subheader('Features Selection', divider = 'rainbow')

    if img_uploaded:
        img_uploaded = Image.open(img_uploaded)
        img_uploaded = img_uploaded.convert('RGB')
        # Show the image preview
        st.image(img_uploaded)

    # Show the features name and selection
    features_selected = st.pills(
        'Select zero or more features that truly describe the image:',
        options = features.keys(),
        selection_mode = 'multi'
    )

    if features_selected:
        # Show the features description
        st.warning(
            'I confirm that these statements below are true:\n\n' + 
            ''.join([ f'- {features[key]}\n' for key in features_selected ])
        )
    else:
        st.error(
            'No feature is selected yet.\n\n **Why is this needed?** These features '
            'will be send to the model along with the image. Not selecting any feature '
            'or faking the features will result in inaccurate feedback/prediction.'
        )
    
    return img_uploaded, features_selected


def result_section(
    model_endpoint: str, img_uploaded: Image.Image | None, features_selected: list
):
    predict = st.button(
        'Predict' if features_selected else 'Predict Anyway',
        type = 'primary' if features_selected else 'secondary',
        disabled = False if img_uploaded else True
    )

    if predict and img_uploaded:
        with st.spinner('Retrieving Prediction...', show_time = True):
            # Convert the image to bytes
            img_bytes = io.BytesIO()
            img_uploaded.save(img_bytes, 'JPEG')

            try:
                response = requests.post(
                    model_endpoint,
                    # This must match the schema provided in the backend (e.g. FastAPI)
                    data = {features: True for features in features_selected},
                    files = {'Image': img_bytes.getvalue()}
                )
            except requests.ConnectionError:
                response = None

        with st.container(border = True):
            st.subheader('Result', divider = 'rainbow')

            if not response:
                st.markdown(
                    'Hmm... Can\'t reach the server to make prediction. If this keeps '
                    'happening, please report this to the administrator.'
                )
            elif response.status_code == 200:
                pred_result = ServeResponse(**response.json())
                pred_result = round(pred_result.result)
                pred_text = f'{pred_result} out of 100'

                if pred_result < 60:
                    pred_text = f'**:red[{pred_text}]**'
                    st.markdown(
                        'Aww... Too bad, but your predicted pawpularity score is only '
                        f'{pred_text}. For the sake of your pet, maybe you should take '
                        'another photo?'
                    )
                elif pred_result <= 80:
                    pred_text = f'**:orange[{pred_text}]**'
                    st.markdown(
                        f'Not bad... Your predicted pawpularity score is {pred_text}, '
                        'but some things certainly can be improved. Perhaps you may '
                        'want to take another photo?'
                    )
                else:
                    pred_text = f'**:green[{pred_text}]**'
                    st.markdown(
                        f'Fantastic! Your predicted pawpularity score is {pred_text}. '
                        'Are you sure you don\'t want to adopt it by yourself? This '
                        'one might become a superstar later...'
                    )
            else:
                st.markdown(
                    'Oops! Something went wrong... Expected response status code 200, '
                    f'but received {response.status_code} ({response.reason}) instead.'
                )

            with st.expander('See raw response'):
                st.write(response.json())


def main():
    dotenv.load_dotenv(
        '.env.prod' if Path('.env.prod').exists() else '.env.dev',
        override = False
    )

    model_endpoint = ServeConf().model_endpoint
    model_fields = ServeRequest.model_fields

    features = {
        # Get all image features and the description
        model_fields[key].alias: model_fields[key].description
        for key in model_fields if model_fields[key].alias != 'Image'
    }

    img_uploaded, features_selected = image_section(features)
    result_section(model_endpoint, img_uploaded, features_selected)


if __name__ == '__main__':
    main()