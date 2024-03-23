from flask import Flask, render_template, request, jsonify
from MLpreparation import MLpreparation as MLpre

# store choise between classification or regression
r_or_c_list = list()
# Whole strings of cost functions report to user 
REPORT_PRINT = list() 

#--------------Start-setup for API-------------#
app = Flask(__name__)

#-------exchange between backend and frontend--------#
@app.route('/')
def index():
    return render_template('index.html') 

# regression or classification
@app.route('/choose-between-r-or-c', methods=['POST'])
def r_or_c():
    data = request.get_json()  
    choice = data.get('choice')
    if choice == 'r':
        r_or_c_list.append('r')
    elif choice == 'c':
        r_or_c_list.append('c')
        # return choice to js->html->to screen
    return jsonify({"success": True, "choice": choice})

# Uppload csv 
@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    if 'csvFile' not in request.files:
        return jsonify({"success": False, "message": "No file part"})
    file = request.files['csvFile']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"})
    if file and file.filename.endswith('.csv'):
        try:
            # Send file to MLpre class for make a dataframe
            df = MLpre(file)
            df.read_csv_pandas()
            # Get basic dataframe info
            info_about_df_to_user = df.info_about_df() 
            return jsonify({"success": True, "message": "File successfully uploaded and read", "df_info": info_about_df_to_user})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    return jsonify({"success": False, "message": "Invalid file type"})

# Show features on the screen
@app.route('/show-csv', methods=['GET'])
def edit_csv():
    # Pick up df
    df = MLpre.send_df()
    if df is not None:
        columns_whole = df.columns
        df_show = ""
        for nr,_ in enumerate(columns_whole):
            df_show += f"{nr}" + " | " + _ + " | " + "\n"   
        return jsonify({"success": True, "df_show": df_show + "\nWhich are your target column?" })
    else:
        return jsonify({"success": False, "message": "No DataFrame available"})
        
# Choose Target
@app.route('/choose-target', methods=['POST'])
def choose_target():
    # if regressor
    if r_or_c_list[-1] == "r":
        # Pick up JSON-data as sent from client
        data = request.get_json()  
        # Get the number from the user
        target_number = data.get('targetNumber')  
        try:
            # pick_up_target_split_and_call_ml, this method do a lot of this
            report_print = MLpre.pick_up_target_split_and_call_ml(int(target_number), "r")
            REPORT_PRINT.append(report_print)
            return jsonify({"success": True, "message": "Number received", "reportText": report_print})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})
    # if classifier
    elif r_or_c_list[-1] == "c":
        data = request.get_json()  
        target_number = data.get('targetNumber') 
        try:
            report_print = MLpre.pick_up_target_split_and_call_ml(int(target_number), "c")
            REPORT_PRINT.append(report_print)
            return jsonify({"success": True, "message": "Number received", "reportText": report_print})
        except Exception as e:
            return jsonify({"success": False, "message": str(e)})

# See models cost function report        
@app.route('/see-report', methods=['POST'])
def go_show_report():
    try:
        # pick up best model
        best_model_score = MLpre.send_best_model_score()
        report = REPORT_PRINT[-1] 
        return jsonify({"success": True, "report": report, "best_model_score": best_model_score})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Dump best model
@app.route('/dump-model-final', methods=['POST'])
def dump_model_final():
    try:
        MLpre.dump_best_model_final() 
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Start Flask App
if __name__ == '__main__':
    app.app_context().push()
    app.run()




