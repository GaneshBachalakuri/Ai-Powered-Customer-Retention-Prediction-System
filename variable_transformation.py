import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import Logger
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, quantile_transform


logger = Logger.get_logs('variable')


def apply_variable_transformations(dummy_x_train, dummy_x_test, X_train, X_test):
    try:
        logger.info("Starting variable transformations")

        # -------------------- 1. Log Transformation --------------------
        log_train = np.log1p(dummy_x_train)
        log_test = np.log1p(dummy_x_test)

        df_log_train = pd.DataFrame(log_train,
                                    columns=[col + '_log' for col in dummy_x_train.columns],
                                    index=dummy_x_train.index)

        df_log_test = pd.DataFrame(log_test,
                                   columns=[col + '_log' for col in dummy_x_test.columns],
                                   index=dummy_x_test.index)

        # -------------------- 2. Square Root Transformation --------------------
        sqrt_train = np.sqrt(dummy_x_train)
        sqrt_test = np.sqrt(dummy_x_test)

        df_sqrt_train = pd.DataFrame(sqrt_train,
                                     columns=[col + '_sqrt' for col in dummy_x_train.columns],
                                     index=dummy_x_train.index)

        df_sqrt_test = pd.DataFrame(sqrt_test,
                                    columns=[col + '_sqrt' for col in dummy_x_test.columns],
                                    index=dummy_x_test.index)

        # -------------------- 3. Standard Scaling --------------------
        std_scaler = StandardScaler()
        std_train = std_scaler.fit_transform(dummy_x_train)
        std_test = std_scaler.transform(dummy_x_test)

        df_std_train = pd.DataFrame(std_train,
                                    columns=[col + '_std' for col in dummy_x_train.columns],
                                    index=dummy_x_train.index)

        df_std_test = pd.DataFrame(std_test,
                                   columns=[col + '_std' for col in dummy_x_test.columns],
                                   index=dummy_x_test.index)

        # -------------------- 4. Min-Max Scaling --------------------
        minmax_scaler = MinMaxScaler()
        mm_train = minmax_scaler.fit_transform(dummy_x_train)
        mm_test = minmax_scaler.transform(dummy_x_test)

        df_mm_train = pd.DataFrame(mm_train,
                                   columns=[col + '_minmax' for col in dummy_x_train.columns],
                                   index=dummy_x_train.index)

        df_mm_test = pd.DataFrame(mm_test,
                                  columns=[col + '_minmax' for col in dummy_x_test.columns],
                                  index=dummy_x_test.index)

        # -------------------- 5. Power Transformation --------------------
        power = PowerTransformer(method='yeo-johnson')
        power_train = power.fit_transform(dummy_x_train)
        power_test = power.transform(dummy_x_test)

        df_power_train = pd.DataFrame(power_train,
                                      columns=[col + '_power' for col in dummy_x_train.columns],
                                      index=dummy_x_train.index)

        df_power_test = pd.DataFrame(power_test,
                                     columns=[col + '_power' for col in dummy_x_test.columns],
                                     index=dummy_x_test.index)

        # -------------------- 6. Quantile Transformation --------------------
        qt_train = quantile_transform(dummy_x_train,
                                      output_distribution='normal',
                                      random_state=42)

        qt_test = quantile_transform(dummy_x_test,
                                     output_distribution='normal',
                                     random_state=42)

        df_qt_train = pd.DataFrame(qt_train,
                                   columns=[col + '_qt' for col in dummy_x_train.columns],
                                   index=dummy_x_train.index)

        df_qt_test = pd.DataFrame(qt_test,
                                  columns=[col + '_qt' for col in dummy_x_test.columns],
                                  index=dummy_x_test.index)

        # -------------------- Merge All Transformations --------------------
        X_train = pd.concat([X_train,
                             df_log_train,
                             df_sqrt_train,
                             df_std_train,
                             df_mm_train,
                             df_power_train,
                             df_qt_train], axis=1)

        X_test = pd.concat([X_test,
                            df_log_test,
                            df_sqrt_test,
                            df_std_test,
                            df_mm_test,
                            df_power_test,
                            df_qt_test], axis=1)

        logger.info("All variable transformations completed successfully")
        logger.info(f"Final Train Shape: {X_train.shape}")
        logger.info(f"Final Test Shape: {X_test.shape}")

        return X_train, X_test

    except Exception:
        exc_type, exc_msg, exc_tb = sys.exc_info()
        logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")
