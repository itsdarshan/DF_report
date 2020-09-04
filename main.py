from df_intent_detection import df_acc_for_curr_phrases, df_acc_for_new_phrases
from config import GU_curr_phrase_dir, GU_new_phrase_dir, GU_curr_phrase_dir_test, \
    GU_report_new_phrase, GU_report_all, GU_report_test, \
    vTech_curr_phrase_dir, vTech_curr_phrase_dir_test, vTech_new_phrase_dir, \
    vTech_report_test, vTech_report_all, vTech_report_new_phrase
import pandas

# Test report for curr intents
report_df_current = df_acc_for_curr_phrases(vTech_curr_phrase_dir_test)
report_df_current_frame = pandas.DataFrame(report_df_current).values

pandas.DataFrame(report_df_current, columns=['intent name', '(failed phrases/total phrases)',
                                             'acc/f1/pr/rec', 'failed phrases']). \
    to_excel(vTech_report_test, index=False, header=True)

# Report related to new phrases
y_curr, y_new = df_acc_for_new_phrases(vTech_curr_phrase_dir_test, vTech_new_phrase_dir)
y_curr_frame = pandas.DataFrame(y_curr).values
y_new_frame = pandas.DataFrame(y_new).values

pandas.DataFrame(y_new_frame, columns=['intent name', '(failed new phrases/total new phrases)',
                                       'acc(%) / f1 / pr / rec', 'new failed phrases']). \
    to_excel(vTech_report_new_phrase, index=False, header=True)

# Main report for curr intents
report_df_all_current = df_acc_for_curr_phrases(vTech_curr_phrase_dir)
report_df_all_current_frame = pandas.DataFrame(report_df_all_current).values

pandas.DataFrame(report_df_all_current, columns=['Intent Name', '(Failed Phrases/Total Phrases)',
                                                 'acc(%) / f1 / pr / rec', 'Failed Phrases']). \
    to_excel(vTech_report_all, index=False, header=True)
