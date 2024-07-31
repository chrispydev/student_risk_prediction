from django.shortcuts import render
from django.views import View
import os
import joblib
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionSerializer
from .scripts.train_model import train_and_save_model


class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        train_and_save_model()
        serializer = PredictionSerializer(data=request.data)
        if serializer.is_valid():
            features = np.array(
                serializer.validated_data['features']).reshape(1, -1)

            # Get the directory of the current script
            main_dir = os.path.dirname(os.path.abspath(__file__))
            script_dir = os.path.join(main_dir, 'scripts')
            model_path = os.path.join(
                script_dir, 'models', 'student_model.pkl')
            print(f'Model path: {model_path}')  # Print the path for debugging

            try:
                # Check if file exists
                if not os.path.isfile(model_path):
                    return Response({'error': 'Model file not found.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

                # Load the model
                model = joblib.load(model_path)

                # Predict
                # prediction = model.predict(features)
                prediction = model.predict(features)[0]
                # Map prediction to a meaningful message
                if prediction == 1:
                    message = "The student is predicted to succeed."
                else:
                    message = "The student is predicted to not succeed."

                return Response({'prediction': message}, status=status.HTTP_200_OK)
                # return Response({'prediction': int(prediction[0])}, status=status.HTTP_200_OK)
            except Exception as e:
                print(f'Error during prediction: {e}')
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class Home(View):
    def get(self, request, *args, **kwargs):
        return render(request, template_name='predictions/index.html')
