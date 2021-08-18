import pickle
import random
import argparse
import numpy as np
import tensorflow as tf

from model import Encoder

global ARGS

def getNumberOfCodes(sets):
  highestCode = 0
  for set in sets:
    for pat in set:
      for adm in pat:
        for code in adm:
          if code > highestCode:
            highestCode = code
  return (highestCode + 1)


def prepareHotVectors(train_tensor, labels_tensor):
  nVisitsOfEachPatient_List = np.array([len(seq) for seq in train_tensor]) - 1
  numberOfPatients = len(train_tensor)
  maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

  x_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float32)
  y_hotvectors_tensor = np.zeros((maxNumberOfAdmissions, numberOfPatients, ARGS.numberOfInputCodes)).astype(np.float32)
  mask = np.zeros((maxNumberOfAdmissions, numberOfPatients)).astype(np.float32)

  for idx, (train_patient_matrix,label_patient_matrix) in enumerate(zip(train_tensor,labels_tensor)):
    for i_th_visit, visit_line in enumerate(train_patient_matrix[:-1]): #ignores the last admission, which is not part of the training
      for code in visit_line:
        x_hotvectors_tensor[i_th_visit, idx, code] = 1
    for i_th_visit, visit_line in enumerate(label_patient_matrix[1:]):  #label_matrix[1:] = all but the first admission slice, not used to evaluate (this is the answer)
      for code in visit_line:
        y_hotvectors_tensor[i_th_visit, idx, code] = 1
    mask[:nVisitsOfEachPatient_List[idx], idx] = 1.

  nVisitsOfEachPatient_List = np.array(nVisitsOfEachPatient_List, dtype=np.int32)

  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  x_hotvectors_tensor = tf.convert_to_tensor(x_hotvectors_tensor, dtype=tf.float32)
  y_hotvectors_tensor = tf.convert_to_tensor(y_hotvectors_tensor, dtype=tf.float32)
  nVisitsOfEachPatient_List = tf.convert_to_tensor(nVisitsOfEachPatient_List, dtype=tf.int32)

  return x_hotvectors_tensor, y_hotvectors_tensor, mask, nVisitsOfEachPatient_List


def loadData():
  main_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  print("-> " + str(len(main_trainSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".train dimensions ")
  main_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))
  print("-> " + str(len(main_testSet)) + " patients at dimension 0 for file: "+ ARGS.inputFileRadical + ".test dimensions ")
  print("Note: these files carry 3D tensor data; the above numbers refer to dimension 0, dimensions 1 and 2 have irregular sizes.")

  ARGS.numberOfInputCodes = getNumberOfCodes([main_trainSet,main_testSet])
  print('Number of diagnosis input codes: ' + str(ARGS.numberOfInputCodes))

  #uses the same data for testing, but disregarding the fist admission of each patient
  labels_trainSet = pickle.load(open(ARGS.inputFileRadical+'.train', 'rb'))
  labels_testSet = pickle.load(open(ARGS.inputFileRadical+'.test', 'rb'))

  train_sorted_index = sorted(range(len(main_trainSet)), key=lambda x: len(main_trainSet[x]))  #lambda x: len(seq[x]) --> f(x) return len(seq[x])
  main_trainSet = [main_trainSet[i] for i in train_sorted_index]
  labels_trainSet = [labels_trainSet[i] for i in train_sorted_index]

  test_sorted_index = sorted(range(len(main_testSet)), key=lambda x: len(main_testSet[x]))
  main_testSet = [main_testSet[i] for i in test_sorted_index]
  labels_testSet = [labels_testSet[i] for i in test_sorted_index]

  trainSet = [main_trainSet, labels_trainSet]
  testSet = [main_testSet, labels_testSet]

  return trainSet, testSet

def buildModel():
  model = Encoder(number_of_codes=ARGS.numberOfInputCodes, encoder_units=ARGS.hiddenDimSize[-1], embedding_dim=300)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
  cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  return model, optimizer, cross_entropy

def evaluateModel(model, optimizer, cross_entropy, test_Set):
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100
  crossEntropySum = 0.0
  dataCount = 0.0

  #computes de crossEntropy for all the elements in the test_Set, using the batch scheme of partitioning
  for index in range(n_batches):
    batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
    batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
    x, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)

    crossEntropy = applyGradient(model, optimizer, cross_entropy, x, y, mask, nVisitsOfEachPatient_List)

    #accumulation by simple summation taking the batch size into account
    crossEntropySum += crossEntropy * len(batchX)
    dataCount += float(len(batchX))
  
  #At the end, it returns the mean cross entropy considering all the batches
  return n_batches, crossEntropySum / dataCount


def applyGradient(model, optimizer, cross_entropy, x, y, mask, nVisitsOfEachPatient_List):
  y = tf.transpose(y, [1,0,2])
  with tf.GradientTape() as tape:
    tape.watch(x)
    predictions = model(x, mask)
    loss_value = cross_entropy(y, predictions)

  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  return loss_value

def trainModel():
  print("==> data loading")
  trainSet, testSet = loadData()

  print("==> model building")
  model, optimizer, cross_entropy = buildModel()

  print ("==> training and validation")
  batchSize = ARGS.batchSize
  n_batches = int(np.ceil(float(len(trainSet[0])) / float(batchSize)))

  bestValidationCrossEntropy = 1e20
  bestValidationEpoch = 0
  bestModelDirName = ''

  iImprovementEpochs = 0
  iConsecutiveNonImprovements = 0
  epoch_counter = 0

  for epoch_counter in range(ARGS.nEpochs):
    iteration = 0
    trainCrossEntropyVector = []
    for index in random.sample(range(n_batches), n_batches):
      batchX = trainSet[0][index*batchSize:(index+1)*batchSize]
      batchY = trainSet[1][index*batchSize:(index + 1)*batchSize]
      x, y, mask, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
      x += np.random.normal(0, 0.1, x.shape)

      loss_value = applyGradient(model, optimizer, cross_entropy, x, y, mask, nVisitsOfEachPatient_List)
      trainCrossEntropyVector.append(loss_value)
      iteration += 1

    print('-> Epoch: %d, mean cross entropy considering %d TRAINING batches: %f' % (epoch_counter, n_batches, np.mean(trainCrossEntropyVector)))
    nValidBatches, validationCrossEntropy = evaluateModel(model, optimizer, cross_entropy, testSet)
    print('      mean cross entropy considering %d VALIDATION batches: %f' % (nValidBatches, validationCrossEntropy))

  # Best results
  print('--------------SUMMARY--------------')
  print('The best VALIDATION cross entropy occurred at epoch %d, the value was of %f ' % (
  bestValidationEpoch, bestValidationCrossEntropy))
  print('Best model file: ' + bestModelDirName)
  print('Number of improvement epochs: ' + str(iImprovementEpochs) + ' out of ' + str(epoch_counter + 1) + ' possible improvements.')
  print('Note: the smaller the cross entropy, the better.')
  print('-----------------------------------')


def evaluationResults():
  batchSize = ARGS.batchSize
  predictedY_list = []
  predictedProbabilities_list = []
  actualY_list = []
  predicted_yList = []
  n_batches = int(np.ceil(float(len(test_Set[0])) / float(batchSize))) #default batch size is 100

  for index in range(n_batches):
    batchX = test_Set[0][index * batchSize:(index + 1) * batchSize]
    batchY = test_Set[1][index * batchSize:(index + 1) * batchSize]
    xf, yf, maskf, nVisitsOfEachPatient_List = prepareHotVectors(batchX, batchY)
    maxNumberOfAdmissions = np.max(nVisitsOfEachPatient_List)

    # Prediction result predicted_y = (TODO)
    predicted_yList.append(predicted_y.tolist()[-1])

    # traverse the predicted results, once for each patient in the batch
    for ith_patient in range(predicted_y.shape[1]):
      predictedPatientSlice = predicted_y[:, ith_patient, :]
      # retrieve actual y from batch tensor -> actual codes, not the hotvector
      actual_y = batchY[ith_patient][1:]
      # for each admission of the ith-patient
      for ith_admission in range(nVisitsOfEachPatient_List[ith_patient]):
        # convert array of actual answers to list
        actualY_list.append(actual_y[ith_admission])
        # retrieves ith-admission of ths ith-patient
        ithPrediction = predictedPatientSlice[ith_admission]
        # since ithPrediction is a vector of probabilties with the same dimensionality of the hotvectors
        # enumerate is enough to retrieve the original codes
        enumeratedPrediction = [codeProbability_pair for codeProbability_pair in enumerate(ithPrediction)]
        # sort everything
        sortedPredictionsAll = sorted(enumeratedPrediction, key=lambda x: x[1], reverse=True)
        # creates trimmed list up to max(maxNumberOfAdmissions,30) elements
        sortedTopPredictions = sortedPredictionsAll[0:max(maxNumberOfAdmissions, 30)]
        # here we simply toss off the probability and keep only the sorted codes
        sortedTopPredictions_indexes = [codeProbability_pair[0] for codeProbability_pair in sortedTopPredictions]
        # stores results in a list of lists - after processing all batches, predictedY_list stores all the prediction results
        predictedY_list.append(sortedTopPredictions_indexes)
        predictedProbabilities_list.append(sortedPredictionsAll)

  # ---------------------------------Report results using k=[10,20,30]
  print('==> computation of prediction results with constant k')
  recall_sum = [0.0, 0.0, 0.0]

  k_list = [10, 20, 30]
  for ith_admission in range(len(predictedY_list)):
    ithActualYSet = set(actualY_list[ith_admission])
    for ithK, k in enumerate(k_list):
      ithPredictedY = set(predictedY_list[ith_admission][:k])
      intersection_set = ithActualYSet.intersection(ithPredictedY)
      recall_sum[ithK] += len(intersection_set) / float(len(ithActualYSet))  # this is recall because the numerator is len(ithActualYSet)

  precision_sum = [0.0, 0.0, 0.0]
  k_listForPrecision = [1, 2, 3]
  for ith_admission in range(len(predictedY_list)):
    ithActualYSet = set(actualY_list[ith_admission])
    for ithK, k in enumerate(k_listForPrecision):
      ithPredictedY = set(predictedY_list[ith_admission][:k])
      intersection_set = ithActualYSet.intersection(ithPredictedY)
      precision_sum[ithK] += len(intersection_set) / float(k)  # this is precision because the numerator is k \in [10,20,30]

  finalRecalls = []
  finalPrecisions = []
  for ithK, k in enumerate(k_list):
    finalRecalls.append(recall_sum[ithK] / float(len(predictedY_list)))
    finalPrecisions.append(precision_sum[ithK] / float(len(predictedY_list)))

  print('Results for Recall@' + str(k_list))
  print(str(finalRecalls[0]))
  print(str(finalRecalls[1]))
  print(str(finalRecalls[2]))

  print('Results for Precision@' + str(k_listForPrecision))
  print(str(finalPrecisions[0]))
  print(str(finalPrecisions[1]))
  print(str(finalPrecisions[2]))

  fullListOfTrueYOutcomeForAUCROCAndPR_list = []
  fullListOfPredictedYProbsForAUCROC_list = []
  fullListOfPredictedYForPrecisionRecall_list = []
  for ith_admission in range(len(predictedY_list)):
    ithActualY = actualY_list[ith_admission]
    nActualCodes = len(ithActualY)
    ithPredictedProbabilities = predictedProbabilities_list[ith_admission]  # [0:nActualCodes]
    ithPrediction = 0
    for predicted_code, predicted_prob in ithPredictedProbabilities:
      fullListOfPredictedYProbsForAUCROC_list.append(predicted_prob)
      # for precision-recall purposes, the nActual first codes correspond to what was estimated as correct answers
      if ithPrediction < nActualCodes:
        fullListOfPredictedYForPrecisionRecall_list.append(1)
      else:
        fullListOfPredictedYForPrecisionRecall_list.append(0)

      # the list fullListOfTrueYOutcomeForAUCROCAndPR_list corresponds to the true answer, either positive or negative
      # it is used for both Precision Recall and for AUCROC
      if predicted_code in ithActualY:
        fullListOfTrueYOutcomeForAUCROCAndPR_list.append(1)
      else:
        fullListOfTrueYOutcomeForAUCROCAndPR_list.append(0)
      ithPrediction += 1

  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
  print("Weighted AUC-ROC score: " + str(metrics.roc_auc_score(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                               fullListOfPredictedYProbsForAUCROC_list,
                                                               average='weighted')))
  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
  PRResults = metrics.precision_recall_fscore_support(fullListOfTrueYOutcomeForAUCROCAndPR_list,
                                                      fullListOfPredictedYForPrecisionRecall_list,
                                                      average='binary')
  print('Precision: ' + str(PRResults[0]))
  print('Recall: ' + str(PRResults[1]))
  print('Binary F1 Score: ' + str(PRResults[2]))  # FBeta score with beta = 1.0
  print('Support: ' + str(PRResults[3]))



def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('inputFileRadical', type=str, metavar='<visit_file>', help='File radical name (the software will look for .train and .test files) with pickled data organized as patient x admission x codes.')
  parser.add_argument('outFile', metavar='out_file', default='model_output', help='Any file directory to store the model.')
  parser.add_argument('--maxConsecutiveNonImprovements', type=int, default=10, help='Training wiil run until reaching the maximum number of epochs without improvement before stopping the training')
  parser.add_argument('--hiddenDimSize', type=str, default='[271]', help='Number of layers and their size - for example [100,200] refers to two layers with 100 and 200 nodes.')
  parser.add_argument('--state', type=str, default='cell', help='Pass cell, hidden or attention to fully connected layer')
  parser.add_argument('--batchSize', type=int, default=100, help='Batch size.')
  parser.add_argument('--nEpochs', type=int, default=1000, help='Number of training iterations.')
  parser.add_argument('--LregularizationAlpha', type=float, default=0.001, help='Alpha regularization for L2 normalization')
  parser.add_argument('--dropoutRate', type=float, default=0.45, help='Dropout probability.')
  parser.add_argument('--learningRate', type=float, default=0.5, help='Learning rate.')

  ARGStemp = parser.parse_args()
  hiddenDimSize = [int(strDim) for strDim in ARGStemp.hiddenDimSize[1:-1].split(',')]
  ARGStemp.hiddenDimSize = hiddenDimSize
  return ARGStemp

if __name__ == '__main__':
  ARGS = parseArguments()

  trainModel()
  evaluationResults()