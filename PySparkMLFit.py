import operator
import argparse

from pyspark.ml import Pipeline  # ПАЙПЛАЙН
from pyspark.ml.evaluation import MulticlassClassificationEvaluator  # ОЦЕНЩИК
from pyspark.ml.feature import VectorAssembler, StringIndexer  # ПРЕОБРАЗОВАНИЕ ДАННЫХ
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator  # ГИПЕРПАРАМЕТРЫ И КРОССВАЛИДАЦИЯ
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier  # МОДЕЛЬ ОБУЧЕНИЯ 1
from pyspark.ml.classification import RandomForestClassifier  # МОДЕЛЬ ОБУЧЕНИЯ 2
from pyspark.ml.classification import GBTClassifier  # МОДЕЛЬ ОБУЧЕНИЯ 3


MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'


def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    # TODO Ваш код
    # ПОДГОТОВКА ДАННЫХ
    train, test = data_path.randomSplit([0.8, 0.2],seed=42)
    indexer_user_type = StringIndexer(inputCol='user_type', outputCol="user_type_index")
    indexer_platform = StringIndexer(inputCol='platform', outputCol="platform_index")

    features = ['user_type_index', 'duration', 'platform_index', 'item_info_events',
                'select_item_events', 'make_order_events', 'events_per_min']
    feature_assembler = VectorAssembler(inputCols=features, outputCol="features")

    # оценщик модели
    evaluator = MulticlassClassificationEvaluator(
        labelCol="is_bot", predictionCol="prediction", metricName="accuracy"
    )

    models_to_evaluate = []

    # 1. Random Forest
    rf_classifier = RandomForestClassifier(labelCol="is_bot", featuresCol="features")
    rf_pipeline = Pipeline(stages=[indexer_user_type, indexer_platform, feature_assembler, rf_classifier])
    rf_paramGrid = ParamGridBuilder() \
        .addGrid(rf_classifier.numTrees, [50, 100]) \
        .addGrid(rf_classifier.maxDepth, [10, 20]) \
        .build()
    rf_cv = CrossValidator(estimator=rf_pipeline, estimatorParamMaps=rf_paramGrid, evaluator=evaluator, numFolds=3)
    models_to_evaluate.append(("Random Forest", rf_cv))

    # 2. Decision Tree
    dt_classifier = DecisionTreeClassifier(labelCol="is_bot", featuresCol="features")
    dt_pipeline = Pipeline(stages=[indexer_user_type, indexer_platform, feature_assembler, dt_classifier])
    dt_paramGrid = ParamGridBuilder() \
        .addGrid(dt_classifier.maxDepth, [5, 10, 15]) \
        .build()
    dt_cv = CrossValidator(estimator=dt_pipeline, estimatorParamMaps=dt_paramGrid, evaluator=evaluator, numFolds=3)
    models_to_evaluate.append(("Decision Tree", dt_cv))

    # 3. Gradient-Boosted Trees
    gbt_classifier = GBTClassifier(labelCol="is_bot", featuresCol="features")
    gbt_pipeline = Pipeline(stages=[indexer_user_type, indexer_platform, feature_assembler, gbt_classifier])
    gbt_paramGrid = ParamGridBuilder() \
        .addGrid(gbt_classifier.maxIter, [20, 50]) \
        .addGrid(gbt_classifier.maxDepth, [5, 10]) \
        .build()
    gbt_cv = CrossValidator(estimator=gbt_pipeline, estimatorParamMaps=gbt_paramGrid, evaluator=evaluator, numFolds=3)
    models_to_evaluate.append(("Gradient-Boosted Trees", gbt_cv))

    # --- ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ И ВЫБОР ЛУЧШЕЙ ---
    best_accuracy = 0
    best_model_name = ""
    best_model = None

    for name, cv_estimator in models_to_evaluate:
        #print(f"Обучение и кросс-валидация {name}...")

        # Обучаем модель с поиском лучших гиперпараметров
        cv_model = cv_estimator.fit(train)

        # Получаем лучшую модель из CV
        current_best_model = cv_model.bestModel

        # Оцениваем её на тесте
        predictions = current_best_model.transform(test)
        accuracy = evaluator.evaluate(predictions)

        #print(f"Точность {name} на тестовой выборке: {accuracy:.4f}\n")

        # Если текущая модель лучше всех предыдущих - запоминаем её
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model = current_best_model

    # Сохраняем абсолютного победителя
    if best_model:
        best_model.write().overwrite().save(model_path)
        #print("--- ЗАВЕРШЕНО ---")



def main(data_path, model_path):
    spark = _spark_session()
    df = spark.read.parquet(data_path)
    process(spark, df, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
