# TunnelData
A repository for data collected in a greenhouse tunnel in Stellenbosch, South Africa. Data includes a training and test set used to train an SVR predictive model. A full data set using over 100 days of data is also included for future model development. 

## Data Structure
The training (Training Set.csv) and test (Test Set.csv) sets are aggregated with solar radiation and outside temperature for the Stellenbosch area. This data was in an hourly resolution, which was then interpolated to 5-minute intervals. This also only includes the middle temperature to be used for modeling of the air temperature inside the tunnel.

The full data set includes all sensor data for 164 days. This data is structured with the front sensor's temperature and humidity, the middle sensor's temperature and humidity, and the back sensor's temperature and humidity. The fan and wet wall state are also included in this.
