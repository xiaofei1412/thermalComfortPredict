# 6小时开发诊断RMSE

含内部真值/未来信息的单因素替换，均不可作为部署或能力门成绩。温度单位°C，能源单位kWh。

| 日期/轨迹 | 诊断替换 | 温度 | 燃气 | 制冷电 | 风机电 |
| --- | --- | ---: | ---: | ---: | ---: |
| 62/excitation | baseline | 0.582663 | 1.089597 | 0.550886 | 0.028186 |
| 62/excitation | initial_PI_oracle | 0.583033 | 1.090343 | 0.551229 | 0.028244 |
| 62/excitation | cooling_mode_oracle | 0.570646 | 1.090554 | 0.552808 | 0.028181 |
| 62/excitation | flow_oracle | 0.625421 | 1.096765 | 0.531613 | 0.009772 |
| 62/excitation | enthalpy_correction_oracle | 0.582855 | 1.089589 | 0.550859 | 0.028263 |
| 62/guarded_rule | baseline | 0.528901 | 1.240074 | 0.522609 | 0.025738 |
| 62/guarded_rule | initial_PI_oracle | 0.529161 | 1.240864 | 0.523032 | 0.025807 |
| 62/guarded_rule | cooling_mode_oracle | 0.517722 | 1.240342 | 0.524787 | 0.025736 |
| 62/guarded_rule | flow_oracle | 0.585362 | 1.239110 | 0.502899 | 0.009815 |
| 62/guarded_rule | enthalpy_correction_oracle | 0.528985 | 1.240122 | 0.522581 | 0.025815 |
| 150/excitation | baseline | 0.487404 | 0.166401 | 1.292570 | 0.102198 |
| 150/excitation | initial_PI_oracle | 0.487504 | 0.167828 | 1.292155 | 0.102411 |
| 150/excitation | cooling_mode_oracle | 0.487317 | 0.166401 | 1.292613 | 0.102248 |
| 150/excitation | flow_oracle | 0.545709 | 0.155129 | 1.174838 | 0.007846 |
| 150/excitation | enthalpy_correction_oracle | 0.487035 | 0.166647 | 1.292555 | 0.101352 |
| 150/guarded_rule | baseline | 0.484290 | 0.036353 | 1.265100 | 0.099868 |
| 150/guarded_rule | initial_PI_oracle | 0.483833 | 0.036606 | 1.264597 | 0.100080 |
| 150/guarded_rule | cooling_mode_oracle | 0.485821 | 0.036353 | 1.264992 | 0.099885 |
| 150/guarded_rule | flow_oracle | 0.521249 | 0.047504 | 1.159056 | 0.007696 |
| 150/guarded_rule | enthalpy_correction_oracle | 0.484308 | 0.036401 | 1.265121 | 0.099032 |
