# Cricket Chirp Temperature Estimation
*Preview*
![preview](https://github.com/user-attachments/assets/72fafccb-f80b-45a7-aeaa-cef71ba16c63)
## Project Overview

This project explores the relationship between **Cricket chirp frequency** and **Temperature in Fahrenheit** to develop a *predictive model* for temperature estimation. Utilizing data from the GLOBE Scientists Blog, the project follows a standard data science workflow encompassing data collection, exploratory data analysis (EDA), visualization, and linear regression modeling. The primary objective is to determine if temperature can be accurately estimated based on cricket chirps.

## Data Source

The dataset used in this analysis is sourced from:
[Measuring temperature using crickets | GLOBE Scientists' Blog](https://www.globe.gov/web/globe-scientists-blog/archived-posts/-/blogs/measuring-temperature-using-crickets)

The original dataset contains 6 columns and 55 rows. This analysis focuses on two key variables: "Chirps 15s" (number of cricket chirps in 15 seconds) and "TempFahrenheit" (temperature in Fahrenheit).

## Methodology

The project methodology involves the Data Science life cycle which uses sequential steps for data processing and model development.

### 1. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is a crucial phase in the data science lifecycle, essential for preparing data for subsequent analysis. This project employed Python with the pandas library to perform data cleaning and identify potential issues such as outliers or missing values, ensuring the data is suitable for analysis.

The EDA process involved the following steps:

1.  **Data Acquisition:** Data was obtained from the source link and saved as a CSV file (`final_project.csv`).

2.  **Environment Setup and Data Loading:**
    * Open your Python IDE you prefer in our project; we will use Spyder.
    * Change your working directory to the location where `final_project.csv` is saved. This option is typically found in the IDE's interface.
      
      ![image](https://github.com/user-attachments/assets/57a4cbb7-14c0-4e11-951c-18a213f85084)

    * Import the pandas library and load the CSV file into a pandas DataFrame: Using the code below.
        ```python
        import pandas as pd
        df = pd.read_csv('final_project.csv')
        ```
        * `import pandas as pd`: Imports the pandas library, commonly aliased as `pd`, which provides data manipulation and analysis tools.
        * `df = pd.read_csv('final_project.csv')`: Reads the data from the 'final\_project.csv' file into a pandas DataFrame. The DataFrame `df` will be used to store and work with the dataset.

3.  **Configuration:** Pandas display options were set to show all columns.
    ```python
    pd.set_option('display.max_columns', None)
    ```
    * `pd.set_option('display.max_columns', None)`: Configures pandas to display all columns in the DataFrame output, preventing truncation for wide datasets.

4.  **Null Value Assessment:** Null values in 'Chirps15s' and 'TempFahrenheit' columns were checked.
    ```python
    sum(df['Chirps15s'].isnull())
    ```
    * We can use the same method to check the null values in the 'TempFahrenheit' column by replacing the column name.
    * `sum(df['Chirps15s'].isnull())`: Calculates the sum of null (missing) values in the 'Chirps15s' column. `isnull()` returns a boolean Series (True for null, False otherwise), and `sum()` counts the True values.
    * In our case, our column doesn't have any null values, so we get output like this:
      
    ![image](https://github.com/user-attachments/assets/5306355b-06aa-4bbe-8036-6227df34a9bb)

  
    Code for dropping null values and updating the dataframe:
    ```python
    df.dropna(subset=['Chirps15s'])
    df = df.dropna(subset=['Chirps15s'])
    ```
    * `df.dropna(subset=['Chirps15s'])`: Creates a new DataFrame by removing rows that have missing (null) values specifically in the 'Chirps15s' column.
    * `df = df.dropna(subset=['Chirps15s'])`: Assigns the new DataFrame above to our current `df`, updating the original DataFrame without any null values for the given column.
    * In our case, we don't need to update the dataframe because we don't have any null values. Therefore, we will skip this step.

5.  **Value Distribution Analysis:** Distinct value counts, minimum, and maximum values were examined for both 'Chirps15s' and 'TempFahrenheit' to identify potential outliers or anomalies.
   
    Check minimum and maximum values:
    ```python
    df['Chirps15s'].min()
    df['Chirps15s'].max()
    df['TempFahrenheit'].min()
    df['TempFahrenheit'].max()
    ```
    * `df['Chirps15s'].min()`: Returns the minimum value in the 'Chirps15s' column.
    * `df['Chirps15s'].max()`: Returns the maximum value in the 'Chirps15s' column.
    * `df['TempFahrenheit'].min()`: Returns the minimum value in the 'TempFahrenheit' column.
    * `df['TempFahrenheit'].max()`: Returns the maximum value in the 'TempFahrenheit' column.

    * 'Chirps15s' min: 12.5, max: 46.4.
   ![image](https://github.com/user-attachments/assets/19113a93-f3ef-4d06-8590-eff1b3aebe24)
    * 'TempFahrenheit' min: 49.25, max: 80.5.
    ![image](https://github.com/user-attachments/assets/72564461-9e4f-4dc5-a409-799df8e8ad92)

        Based on our source information, temperatures below 50°F and potentially very high temperatures may correlate with less reliable chirp data.
    Since we are still not sure of the range, but we want to find the values below or above a range, we can use the Code for filtering based on value ranges:
    ```python
    # Assuming values below the provided value are outliers
    df[df['Chirps15s'] < 15]
    ```
    * `df[df['Chirps15s'] < 15]`: Selects and displays rows from the DataFrame where the value in the 'Chirps15s' column is less than 15.
      
    ![image](https://github.com/user-attachments/assets/d60a88dc-6443-4390-affb-9d21413523a2)
    ```python
    # Assuming the values above the provided value are outliers
    df[df['Chirps15s'] > 40]
    ```
    * `df[df['Chirps15s'] > 40]`: Selects and displays rows from the DataFrame where the value in the 'Chirps15s' column is greater than 40.
      
    ![image](https://github.com/user-attachments/assets/0fb38f44-0a0d-4d18-bc8b-d433b1bdf46c)
    But we can also combine them and find values between those ranges using the code:
    ```python
    df[(df['Chirps15s'] > 14) & (df['Chirps15s'] < 41)]
    df = df[(df['Chirps15s'] > 14) & (df['Chirps15s'] < 41)] # To update DataFrame
    ```
    * `df[(df['Chirps15s'] > 14) & (df['Chirps15s'] < 41)]`: Selects and displays rows where 'Chirps15s' is greater than 14 AND less than 41, using boolean indexing and the logical AND operator (`&`).
    * `df = df[(df['Chirps15s'] > 14) & (df['Chirps15s'] < 41)]`: Updates the DataFrame `df` to contain only the rows where 'Chirps15s' is within the specified range.
      
    ![image](https://github.com/user-attachments/assets/bc01015a-1613-4eb1-900d-ac5df9aed59d)

    * But in our project, we will skip this step and only update our Dataframe with the two columns we need, `Chirps15s` and `TempFahrenheit`, which we will see in our next step.

    

7.  **Feature Selection:** The DataFrame was subsetted to include only 'Chirps15s' and 'TempFahrenheit'.
    ```python
    df = df[['Chirps15s', 'TempFahrenheit']]
    ```
    * `df[['Chirps15s', 'TempFahrenheit']]`: Creates a new DataFrame containing only the `Chirps15s` and `TempFahrenheit` columns from the original `df`.
    * `df = df[['Chirps15s', 'TempFahrenheit']]`: Assigns the above new subsetted DataFrame back to our `df`, keeping only the above two columns for further analysis.

8.  **Descriptive Statistics:** Summary statistics to understand the central tendency and dispersion.
    ```python
    df.describe()
    ```
     * `df.describe()`: Generates descriptive statistics for the numerical columns in the DataFrame `df`, including count, mean, standard deviation, minimum, maximum, and quartile values (25%, 50%, 75%).
     * The 25th percentile (lowest interquartile range) and the 75th percentile (highest interquartile range) values help identify potential outliers because values outside this range are considered potential outliers.
     * Since we are not going to filter our data without any outliers, we will not worry much about the interquartile range in our exploratory analysis.

      ![image](https://github.com/user-attachments/assets/b7bfddf9-2060-4027-999b-50ed6f3b557b)


    Get descriptive statistics for individual columns:
    ```python
    df['Chirps15s'].describe()
    df['TempFahrenheit'].describe()
    ```
    * `df['Chirps15s'].describe()`: Generates descriptive statistics specifically for the 'Chirps15s' column.
    * The 25th percentile is approximately 22.38 (rounded), and the 75th percentile is 35.00. Values outside this range are considered potential outliers.
      
      ![image](https://github.com/user-attachments/assets/683cec68-5e92-4b82-aa4e-e9ec16225c90)
      


    * `df['TempFahrenheit'].describe()`: Generates descriptive statistics specifically for the 'TempFahrenheit' column.
    * The 25th percentile is 60.50, and the 75th percentile is 71.50. Values outside this range are considered potential outliers.
      
      ![image](https://github.com/user-attachments/assets/4e2e7b0c-89fc-45c3-8319-d45b30769729)


9.  **Data Export:** The cleaned and subsetted data were saved to a new CSV file (`Updated_final.csv`) for subsequent analysis in KNIME.
    ```python
    df.to_csv('Updated_final.csv')
    ```
    * `df.to_csv('Updated_final.csv')`: Writes the contents of the DataFrame `df` to a new CSV file named 'Updated\_final.csv' and saves it in our current working directory.

#### 10. Visualization

We will import our `Updated_final.csv` into the KNIME Analytics Platform to visually assess the relationship between the `Chirps15s` and `TempFahrenheit` columns.

**KNIME Workflow Steps:**

1.  Create a new project and name it as your preference, but for our case, we will name it `final_project`.
2.  Add the CSV Reader node. You can also find the CSV reader under `IO → Read → CSV Reader →` drag and drop, or double click to add to your project.  
   ![image](https://github.com/user-attachments/assets/ec181bb8-f5f0-4f73-9226-a63331bf3d76)
4.  Then right-click on the CSV Reader node that you have added in your project and choose Configure.
   ![image](https://github.com/user-attachments/assets/0eca6709-c6ff-40e4-b6f4-3aafe5ab5bf2)

6.  In the configuration dialog box, find your file using the Browse option.
   ![image](https://github.com/user-attachments/assets/4e01655b-a636-4948-957b-c33c5b384695)

8.  Check the Has column header and Has RowID because our updated file has those headers and row ID. Click Apply or Ok and execute it.
   ![image](https://github.com/user-attachments/assets/06ba20f3-ddf4-4135-8e77-bcb2e40b2bca)

10.  Add the `Scatter Plot` node from the Nodes. We can also use:  
    Views ➡️ Visualization Column Appender ➡️ Scatter Plot ➡️ Insert using drag or double click

12.  Connect the `CSV Reader` node to a `Scatter Plot` node.  
    ![image](https://github.com/user-attachments/assets/a4036d93-5f7c-4b74-b8de-0f7360b45967)  ![image](https://github.com/user-attachments/assets/ee42c54d-2ba9-4c5e-a0b5-2491311d15f2)


14.  The three dots at the bottom of nodes indicate status: red (needs configuration), yellow (not executed), green (executed).  
   ![image](https://github.com/user-attachments/assets/ed93a97f-f8f4-4308-892d-aa0803a8524a)
![image](https://github.com/user-attachments/assets/7494abd3-e124-4033-83e2-7985fbca4794)![image](https://github.com/user-attachments/assets/5b963fa7-3f36-42d5-bf69-80566ff35962)
    

16.  Configure the Scatter Plot node: assign 'Chirps15s' to the x-axis and 'TempFahrenheit' to the y-axis. Execute the node and view the plot.  
    Steps ➡️ Right-click ➡️ Configure ➡️ Horizontal: Chirps15s ➡️ Vertical: TempFahrenheit ➡️ Ok and Excute   
    ![image](https://github.com/user-attachments/assets/08fa4fae-944c-4f06-aa08-245b4796095f)
    ![image](https://github.com/user-attachments/assets/dd38275d-0af7-4ed9-b628-e73443deea00)

18.   We can also use the preview box or manually open the scatter plot on our screen using:  
    Right click Scatter plot node ➡️ Open View ➡️ Preview pops up, or we can also use the Open in new window option
     ![image](https://github.com/user-attachments/assets/d08f1ee8-b2e6-4cc6-8d64-b45e16638973)

    

The scatter plot demonstrated a clear positive linear relationship, indicating that as temperature increases, the number of cricket chirps in 15 seconds also increases.

### Refining the Question

Now let's look back into our `exploratory data analysis` and examine our findings and determine if we need to change our `original question`, "Can the outside temperature be estimated by the frequency of cricket chirps?" Based on our current findings, the raw data, its ranges, potential outliers, and visual representation in our scatterplot, there is a relationship that exists relevant to the question. If the raw data had missing values, and the scatterplot had shown `no pattern` or `non-linear relationship`, we could have concluded that this specific data is not sufficient to answer our question. Therefore, we will proceed with our question using the current data because we have a visible `positive linear relationship` between `“Chirps15s”` and `“TempFahrenheit”` data.

### 3. Model Building

With the `positive linear relationship` confirmed, we will use KNIME for the next step: building a `supervised learning model` trained on our data to predict `temperature` from unknown `chirp counts`.

**KNIME Workflow Steps:**

1.  Connect the CSV Reader node to a Linear Correlation node.
    * The Linear Correlation node calculates the correlation coefficient between selected numerical columns, indicating the strength and direction of their linear relationship.  
      ![image](https://github.com/user-attachments/assets/a2245a2e-ae2e-4876-ad44-281da261f033)

2.  Configure the Linear Correlation node to include `'Chirps15s'` and `'TempFahrenheit'`. Execute the node.  
   ![image](https://github.com/user-attachments/assets/bfae10bb-6588-49a3-94a1-ca6880e06f2c)  

    * Next right-click Linear Correlation and ➡️ Open output port ➡️ Under Correlation measure `Table`
    * You can also always use the preview section to jump between `Table` and `Statistics` for efficiency.  
   ![image](https://github.com/user-attachments/assets/0fce1253-65e3-40eb-ac9d-574653bc2ae6)

    * The resulting correlation coefficient is 0.98, confirming a strong positive linear association.
    * In general, the correlation coefficient has to be close to a positive one to hold a strong relationship.
    
4.  Connect the CSV Reader node to a Linear Regression Learner node.
    * The Linear Regression Learner node trains a linear regression model to find the best-fitting linear relationship between a target variable and one or more predictor variables.  
      ![image](https://github.com/user-attachments/assets/743a7880-f915-4c4b-a2a0-a4a72aed8c61)  

5.  Configure the Linear Regression Learner node: set `'TempFahrenheit'` as the `target variable` and `'Chirps15s'` as the `included` values. Execute the node.  
   ![image](https://github.com/user-attachments/assets/fbb0f758-6b67-440b-81f8-c4fc62d9f85e)  
    * Right-Click Linear Regression Learner node ➡️ Open output port ➡️ Under Coeffecients and Statistics `Table`  
   ![image](https://github.com/user-attachments/assets/0e947bc2-e046-4a44-b95a-d6696a6d2ed0)
    * We can see the required values that will help us manually find the `Temperature` from the given `Chirps` counts.  
   ![image](https://github.com/user-attachments/assets/74040d3f-9899-4fb7-9153-b7f801aace6d)

    * Under the Coeff. Column the `Chrips15s` value is our `"m"` value, and `Intercept` is our `"b"` value. The `Chirps15s` value is our `slope`, known as `"m"`, and the `intercept` value is our `y-intercept` value, known as `"b"` in our linear equation, which is:  
`y = mx + b`

    * The value `“x”` will be the one that we provide as `Chirps count` to the model, which will help us find the `“y”` value that is `TempFahrenheit`.
    * Now our linear equation that will predict the TempFahrenheit value will be ► `y = 0.983 x + 40.025`
    * We can also call this as our line of best fit to make a prediction for the `TempFahrenheit` value.
    * Now we will predict a few values to test our model. We can create a new CSV file with the same column name as `“Chirps15s”` and assign some random values to predict `“TempFahrenheit”` values, or we can add a `Table Creator` node in KNIME and assign the same column name and values that we want to provide.
    * In our case, we will add a `Table Creator` node. Search for the Table Creator node and add it to your project.  
      Right-click to configure ➡️ Left Double Click on the `column header` to open properties ➡️ Change the `column name and type`
      ![image](https://github.com/user-attachments/assets/099120aa-e298-4967-94dd-2e1b18b606fd)![image](https://github.com/user-attachments/assets/253159d5-a18a-4d83-a77c-29d0066c197f)
    * If we were to create a `new CSV file`, then we would have to use `another CSV reader` node to `import` and `read` that file.
    

7.  Regression Predictor node:
   * The `Regression Predictor` node applies a trained regression model (from the Learner node) to new data to generate predictions for the target variable.
   * Add the `Regression Predictor` node and connect it with `Linear Regression Learner` and the new CSV file, or in our case, the `Table Creator`, to predict the values.  
     ![image](https://github.com/user-attachments/assets/328fefca-a2c3-4949-bf4a-88d10378ca57)
   * It will then output the predicted values of the `TemFahrenheit` with the provided values of `Chirps15s`.  
     ![image](https://github.com/user-attachments/assets/cf74ae92-24c4-49ab-a5a7-6a5dfd516df4)



**Manual Prediction Example:**

Using our developed `linear equation`, the estimated `temperature` for `40 cricket chirps` in `15 seconds` is calculated as follows:
```
y = 0.983 * (40) + 40.025  # Replacing the 'x' with '40'
y = 39.32 + 40.025  
y = 79.345             
```
Estimated temperature: ~79.35°F.

### **Interpretation and Conclusion**  
Finally, in our project, we used the `data science life cycle` to determine if outside `temperature` can be estimated by cricket `chirp frequency`. We used a data collection method to gather data from the source and perform an `ETL process` to clean the data using pandas. We also used an `exploratory data analysis` technique, including `scatterplot` visualization, to analyze the `relationship` between `Chirp count 15 seconds` and the `temperature in Fahrenheit` values. This revealed a `strong positive linear relationship`, a correlation coefficient of `0.98`. This confirmed that the data is `suitable to build` our `supervised linear regression model`, which can be used to `predict` the estimated `temperature` based on cricket `chirp count`. With the help of our model, we predicted a few temperature values for chirp counts, and further, we manually calculated the temperature based on `40 chirps count` using our `linear equation`.

License:    
This project is licensed under the **MIT License**.
