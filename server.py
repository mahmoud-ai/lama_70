from flask import Flask, request, jsonify, render_template
from lama_chat_hf import generate_text, create_dialouge
#from lama_chat_lc import generate_text #, create_dialouge


app = Flask(__name__)

    
@app.route("/run-lama", methods=["POST"]) 
def get_form_data():
    try:
        crisis = request.json.get("crisis")
        sectors = request.json.get("sectors")
        is_injuries = request.json.get("is_injuries")
        #model = request.json.get("model")

        response = {'data':{}}
             
        result = get_response(create_dialouge(crisis= crisis, sector= sector, is_injuries=is_injuries))
        response["data"][sector] = result
        return response


    
    except Exception as e:
    
        print(f"An exception occurred: {e}")

        return {'data': None }





if __name__ == "__main__":

    app.run(port=5000)
