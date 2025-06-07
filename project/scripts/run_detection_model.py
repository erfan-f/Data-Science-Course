from train_model import train_detection_model
from make_prediction import predict_on_test_set, evaluation_on_test_set

if __name__ =='__main__':
    print("\n🚀 Starting model training...")
    df_test = train_detection_model(oTest=True)

    print("\n📊 Evaluating model on test set...")
    evaluation_on_test_set(df_test)
    
    print("\n🔍 Making predictions on test set...")
    predict_on_test_set(df_test)
