# sudo-solo

The growing demand for renewable energy highlights the need for efficient and reliable operation of solar farms. However, the efficiency of solar panel systems often decreases due to faults and a lack of cost effective cleaning mechanisms. Manual cleaning requires extensive human resources, time and water, while faults such as micro-cracks, hotspots, delamination and diode failures further degrade system performance.



To overcome these limitations, we propose a GenAI based fault detection and resolution system with automated cleaning powered by a cleaning robot that uses effective water conservation method, which automates monitoring, diagnosis and maintenance decision making in large scale solar farms.



The proposed solution integrates real time sensor data including electrical and thermal parameters of solar panels with environmental data such as temperature, humidity, particulate concentration and luminance. These data are transmitted to the control unit through networking protocols. The image acquisition subsystem operates in two modes: (1) autonomous drones that are deployed either on schedule or when triggered by environmental particle data and (2) cameras mounted on an automated cleaning bot that captures panel images during cleaning. The images from the acquisition system and the sensor data are sent to control unit.



# Architecture Overview:

1. Data Layer
2. Communication Layer
3. Processing Layer
4. Application Layer
5. Action Layer


Therefore, the processing unit receives the data and feeds them to the developed AI model. Sensor data are analyzed against the ideal operating condition and diagnostic reports will be generated. Meantime, the optimal cleaning schedules will be determined and evaluated. Simultaneously, image data are processed using the trained fault detection algorithms under the AI model to identify and classify issues. The prototype is built using Raspberry pi with the sensors, camera and the GenAI model expressed high accuracy and strong predictive capability for early fault detection along with the effective performance of Cleaning Robot.
