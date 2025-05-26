# src/detectors/STUDD/student_teacher.py
import numpy as np
import pandas as pd
from skmultiflow.drift_detection.adwin import ADWIN
from src.models.utils import get_model_instance


class StudentTeacherDriftDetector:

    def __init__(self, student_model_name, adwin_delta):
        """
        Student-Teacher Drift Detector.
        Implements a drift detection method where a Student model mimics a Teacher model.
        Drift is detected when the prediction disagreement (mimicking error) increases, as measured by ADWIN

        Args:
            student_model_name (str): Name of the sklearn-compatible model used for the Student.
            adwin_delta (float, optional): Confidence parameter for the ADWIN drift detector. Defaults to 0.002.
        """
        self.student_model_name = student_model_name
        self.student = get_model_instance(self.student_model_name)
        self.adwin = ADWIN(delta=adwin_delta)
        self.drift_detected = False
        self.feature_indices = None  # SHAP Enhancement: Indices for student training

    def set_feature_indices(self, indices):
        """
        Sets the feature indices used by the Student model (SHAP-style filtering).

        Args:
            indices (list[int] or None): List of indices to use as input features for the Student. If None, use all.
        """

        self.feature_indices = indices
        print(f"ST Detector - Student using feature indices: {self.feature_indices}")

    def update_student(self, main_teacher_model, X_data):
        """
        Trains or retrains the Student model using Teacher predictions on the provided data.
        Resets ADWIN after training.

        Args:
            main_teacher_model (sklearn classifier): The trained Teacher model with a `.predict` method.
            X_data (array-like): Input data used to train the Student on Teacher's predictions.
        """

        print(f"ST Detector: updating Student...")
        self.student_is_trained = False

        # Ensure X_data is numpy array
        if isinstance(X_data, pd.DataFrame):
            X_data_np = X_data.values
        elif isinstance(X_data, list):
            X_data_np = np.array(X_data)
        elif isinstance(X_data, np.ndarray):
            X_data_np = X_data
        else:
            raise ValueError("X_data must be either a pd.DataFrame or a list")

        if X_data_np.shape[0] == 0:
            raise ValueError("X_data must not be empty")

        # Get predictions from the Teacher
        y_hat_teacher = main_teacher_model.predict(X_data_np)

        unique_classes = np.unique(y_hat_teacher)
        if unique_classes.size < 2:
            print(f"Warning: Student update skipped – teacher predictions only contain {unique_classes.size} classes")
            return

        # Prepare filtered data for the Student
        X_student_data = X_data_np
        if self.feature_indices is not None:
            X_student_data = X_data_np[:, self.feature_indices]

        # Fit the Student model
        self.student.fit(X_student_data, y_hat_teacher)

        # Reset ADWIN detector
        self.adwin = ADWIN(delta=self.adwin.delta)

    def process_instance(self, main_teacher_model, X_instance):
        """
        Processes a single instance through Teacher and Student models, computes
        mimicking error, and updates ADWIN.

        Args:
            main_teacher_model (sklearn classifier): Trained Teacher model with `predict_proba`.
            X_instance (array-like): New instance to evaluate.

        Returns:
            bool: True if drift was detected by ADWIN, False otherwise.
        """

        self.drift_detected = False

        # Ensure instance is 2D numpy array
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_numpy()
        if X_instance.ndim == 1:
            X_instance_2d = X_instance.reshape(1, -1)
        elif isinstance(X_instance, pd.DataFrame):
            X_instance_2d = X_instance.values
        else:
            X_instance_2d = X_instance

        # Teacher probability prediction
        probs_teacher = main_teacher_model.predict_proba(X_instance_2d)[0]
        y_hat_teacher_prob = np.max(probs_teacher)
        teacher_pred_class_idx = np.argmax(probs_teacher)

        # Student probability prediction
        X_student_instance = X_instance_2d
        if self.feature_indices is not None:
            X_student_instance = X_instance_2d[:, self.feature_indices]
        probs_student = self.student.predict_proba(X_student_instance)[0]

        # Get student's probability for the class the teacher predicted
        if teacher_pred_class_idx < len(probs_student):
            y_hat_student_prob = probs_student[teacher_pred_class_idx]
        else:
            y_hat_student_prob = 0.0

        # Calculate mimicking error
        student_error = np.abs(y_hat_teacher_prob - y_hat_student_prob)

        # Add error to ADWIN
        self.adwin.add_element(student_error)
        if self.adwin.detected_change():
            self.drift_detected = True
            print(f"ST Drift Detected! ADWIN error: {student_error:.4f}")
        return self.drift_detected
