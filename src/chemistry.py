
import PyCO2SYS as pyco2

def compute_ions(y_pred, X_original):
    results = pyco2.sys(
        par1=y_pred,
        par2=8.1,
        par1_type=4,
        par2_type=3,
        salinity=X_original['Salinity'],
        temperature=X_original['Temperature']
    )
    return results