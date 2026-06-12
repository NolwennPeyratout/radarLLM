import json


def find_path(output_file, val_file, qa_id_to_find):
    with open(output_file, "r") as f:
        predicted_captions = json.load(f)
    
    for element in predicted_captions:
        if element["qa_id"] == qa_id_to_find:
            print(f"Found qa_id {qa_id_to_find} in output file.")
            print(f"Predicted caption: {element['caption']}")
            break

    with open(val_file, "r") as f:
        val_data = json.load(f)

    for item in val_data:
        if item["id"] == qa_id_to_find:
            print(f"Found qa_id {qa_id_to_find} in validation set.")
            print(f"Question: {item['question']}")
            print(f"Answer: {item['answer']}")
            print(f"Sample token: {item['sample_token']}")
            break


if __name__ == "__main__":

    qa_id_to_find = 28657
    output_file = "/home/renault/repo/radarllm/lavis/output/radarLLM/radar_captioning/20260515100_with_adapter/result/val_epoch1.json"
    val_file = "/home/renault/repo/radarllm/dataset/LiDAR-LLM-Nu-Caption/val.json"
    find_path(output_file, val_file, qa_id_to_find)