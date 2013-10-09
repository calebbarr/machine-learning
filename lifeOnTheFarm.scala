/**
@author Caleb Barr
Fun little script to do logistic regression classification
very succinctly using the in-memory, distributed framework Spark.
**/

// set threshold and scaling factor parameters
val threshold = .5; val scalingFactor = 1
// open text file, split into labels and docs, not cached, so will be recomputed when needed and not take up memory
val lines = sc.textFile("farm.txt").map(_.split("\\s+",2))
// get feature space
val allFeatures = lines.map { array => array(1) }.flatMap(_.split("\\s+")).distinct
// get labeled feature vectors
val instances = lines map { array => (array(0).toDouble, { val features = Set()++array(1).split("\\s+"); new spark.util.Vector(allFeatures.map { f => if(features.contains(f)) 1.0 else 0.0 }.collect ) } ) }
// mark RDD for caching, compute it (to avoid strange NullPointerExceptions later)
instances.cache; instances.first
// compute initial random hyperplane
var w = new spark.util.Vector( (allFeatures map { _:String => 2 * new java.util.Random(42).nextGaussian * scalingFactor }).collect )
// train model
for(i <- 0 until 12) w -= instances.map { p => (1 / (1 + math.exp(-p._1 * (w dot p._2))) - 1) * p._1 * p._2 } reduce(_ + _)
// initialize metric containers
var percentCorrect,precisionPositive,precisionNegative,weightedAvgPrecision,recallPositive,recallNegative,weightedAvgRecall,proportionPositive,proportionNegative = 0.0
val truePositives,falsePositives,trueNegatives,falseNegatives = sc.accumulator(0.0)
// logit link function to map linear regression of log-odds to probability space
def logit = {(dotProduct:Double) =>  { val maxAbsoluteValueForExponentiation = 709.7825; if(math.abs(dotProduct) < maxAbsoluteValueForExponentiation)
          (math.exp(dotProduct) / (1 + math.exp(dotProduct) )) else if(dotProduct < 0)  0.0 else  1.0 } }
// classify instances, evaluate each classification
instances map { instance => { (instance._1, instance._2.dot(w) > threshold )}} foreach { case(label,classification) => { (label,classification) 
  match {
  case (1.0,true) => {truePositives +=1} 
  case (1.0,false) => { falsePositives +=1} 
  case (-1.0,false) => {trueNegatives +=1}  
  case (-1.0,true) => {falseNegatives += 1} } } }
// get metrics
percentCorrect = (truePositives.value + trueNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value)
proportionPositive = (truePositives.value + falseNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value)
proportionNegative = (falsePositives.value + trueNegatives.value) / (truePositives.value + trueNegatives.value + falsePositives.value + falseNegatives.value)
precisionPositive = truePositives.value / (truePositives.value + falsePositives.value)
precisionNegative = trueNegatives.value / (trueNegatives.value + falseNegatives.value)
weightedAvgPrecision = (precisionPositive * proportionPositive) + (precisionNegative * proportionNegative)
recallPositive = truePositives.value / (falseNegatives.value + truePositives.value)
recallNegative = trueNegatives.value / (falsePositives.value + trueNegatives.value)
weightedAvgRecall = (precisionPositive * proportionPositive) + (precisionNegative * proportionNegative)
