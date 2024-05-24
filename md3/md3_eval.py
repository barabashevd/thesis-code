from sklearn.metrics import confusion_matrix, f1_score
import bodmas.multiple_evaluate as evaluate
import bodmas.utils as utils
import os
import logging


class MD3Evaluator:
    def __init__(self, report_path, report_folder, test_begin_time, test_end_time, seed):
        self.report_path = report_path
        self.report_folder = report_folder
        self.test_begin_time = test_begin_time
        self.test_end_time = test_end_time
        self.seed = seed

    def evaluate_classifier(self, model, X_val, X_test, y_val, y_test, test_month, POSTFIX):
        fpr_list_all, tpr_list_all, f1_list_all = [], [], []  # each has two lists (fpr threshold 0.01 and 0.001)
        fpr_true_list, tpr_true_list, f1_true_list = [], [], []

        for (x, y) in [(X_val, y_val), (X_test, y_test)]:
            # Make predictions
            y_val_pred = model.predict(x)
            # Calculate the confusion matrix
            y_pred = (y_val_pred >= 0.5).astype(int)
            cm = confusion_matrix(y, y_pred)
            TN, FP, FN, TP = cm.ravel()

            fpr_true_list.append(FP / (FP + TN))
            tpr_true_list.append(TP / (TP + FN))
            f1_true_list.append(f1_score(y, y_pred))

        clf = 'gbdt'

        target_fpr = [0.01, 0.001]

        for fpr_target_on_val in target_fpr:
            threshold, fpr, tpr, f1 = evaluate.evaluate_prediction_on_validation(model, X_val, y_val,
                                                                                 fpr_target_on_val,
                                                                                 model_name=clf)

            fpr_list = [fpr]
            tpr_list = [tpr]
            f1_list = [f1]

            logging.critical(f'validation set threshold: {threshold}')

            phase = test_month

            ALL_CLASSIFICATION_RESULT_PATH = os.path.join(self.report_folder, 'intermediate',
                                                          f'{clf}_test_fpr{fpr_target_on_val}_all_classification_result_{POSTFIX}.csv')
            utils.create_parent_folder(ALL_CLASSIFICATION_RESULT_PATH)
            MISCLASSIFIED_RESULT_PATH = os.path.join(self.report_folder, 'intermediate',
                                                     f'misclassified_{clf}_test_fpr{fpr_target_on_val}_result_{POSTFIX}.csv')

            fpr, tpr, f1 = evaluate.evaluate_prediction_on_testing(model, phase, X_test, y_test, threshold,
                                                                   self.test_begin_time, self.test_end_time, self.seed,
                                                                   ALL_CLASSIFICATION_RESULT_PATH,
                                                                   MISCLASSIFIED_RESULT_PATH,
                                                                   model_name=clf, detail=False)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            f1_list.append(f1)

            fpr_list_all.append(fpr_list)
            tpr_list_all.append(tpr_list)
            f1_list_all.append(f1_list)

        # add roc_auc_score for a fair comparison
        auc_score_list = evaluate.evaluate_auc_score(model, clf, X_val, y_val, [X_test], [y_test])

        # Check if the file exists and its size
        file_exists = os.path.exists(self.report_path)
        file_is_empty = not os.path.getsize(self.report_path) > 0 if file_exists else True

        with open(self.report_path, 'a') as f:
            if file_is_empty:
                # This means the file is newly created or empty, so write the header
                f.write(
                    'phase,fpr_0.1%,tpr_0.1%,f1_0.1%,fpr_0.01%,tpr_0.01%,f1_0.01%,fpr_true,tpr_true,f1_true,auc_score\n')

            phase_list = [f'val_{test_month}'] + [f'test_{test_month}' for _ in range(len(auc_score_list) - 1)]

            for i in range(len(phase_list)):
                phase = phase_list[i]
                fpr_1 = fpr_list_all[0][i]
                fpr_2 = fpr_list_all[1][i]
                fpr_3 = fpr_true_list[i]
                tpr_1 = tpr_list_all[0][i]
                tpr_2 = tpr_list_all[1][i]
                tpr_3 = tpr_true_list[i]
                f1_1 = f1_list_all[0][i]
                f1_2 = f1_list_all[1][i]
                f1_3 = f1_true_list[i]
                auc_score = auc_score_list[i]

                f.write(f'{phase},'
                        f'{fpr_1 * 100:.4f}%,{tpr_1 * 100:.2f}%,{f1_1 * 100:.2f}%,' +
                        f'{fpr_2 * 100:.4f}%,{tpr_2 * 100:.2f}%,{f1_2 * 100:.2f}%,' +
                        f'{fpr_3 * 100:.4f}%,{tpr_3 * 100:.2f}%,{f1_3 * 100:.2f}%,' +
                        f'{auc_score * 100:.4f}%\n')

    def write_result_to_report(self, fpr_list, tpr_list, f1_list, auc_score_list, report_path, test_month):
        phase_list = ['val'] + [f'test_{test_month}' for i in range(len(auc_score_list) - 1)]

        # Check if the file exists and its size
        file_exists = os.path.exists(report_path)
        file_is_empty = not os.path.getsize(report_path) > 0 if file_exists else True

        with open(report_path, 'a') as f:
            if file_is_empty:
                # This means the file is newly created or empty, so write the header
                f.write('phase,fpr_0.1%,tpr_0.1%,f1_0.1%,fpr_0.01%,tpr_0.01%,f1_0.01%,auc_score\n')

            for i in range(len(phase_list)):
                phase = phase_list[i]
                fpr_1 = fpr_list[0][i]
                fpr_2 = fpr_list[1][i]
                tpr_1 = tpr_list[0][i]
                tpr_2 = tpr_list[1][i]
                f1_1 = f1_list[0][i]
                f1_2 = f1_list[1][i]
                auc_score = auc_score_list[i]
                f.write(f'{phase},{fpr_1 * 100:.4f}%,{tpr_1 * 100:.2f}%,{f1_1 * 100:.2f}%,' +
                        f'{fpr_2 * 100:.4f}%,{tpr_2 * 100:.2f}%,{f1_2 * 100:.2f}%,{auc_score * 100:.4f}%\n')

        logging.info(f'write result to {report_path} done')
