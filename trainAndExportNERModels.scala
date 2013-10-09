/**
 * @author Caleb Barr
A routine to train NER models using OpenNLP and export them to S3.
**/

package cbarr.ml.ner

import java.nio.charset.Charset
import opennlp.tools.util.PlainTextByLineStream
import java.io.FileInputStream
import opennlp.tools.namefind.NameSampleDataStream
import opennlp.tools.namefind.TokenNameFinderModel
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.namefind.TokenNameFinderEvaluator
import opennlp.tools.util.TrainingParameters
import java.util.Collections
import scala.collection.mutable.HashMap
import java.io.ByteArrayInputStream
import java.io.File
import java.util.Scanner
import scala.collection.mutable.MutableList
import opennlp.tools.util.model.BaseModel
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.FileWriter
import com.amazonaws.services.s3.AmazonS3Client
import com.amazonaws.auth.BasicAWSCredentials
import opennlp.tools.cmdline.namefind.TokenNameFinderDetailedFMeasureListener
import  opennlp.tools.namefind.TokenNameFinder


object TrainNERModels {
  val DATA_SET = "REFLEX_ENG"
  val LOG_FILENAME = "NER_status.txt"
  
  // make sure the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set if used
  val S3_EXPORT = true  
  val BUCKET_NAME = "cbarr"
  val REMOTE_AWS_DIR = "NER"
  
  val CHARSET = "UTF-8"
  val entityCounts = HashMap[String,Integer]()
  /**
   *  @args: input directory
   */
  def main(args: Array[String]): Unit = {
  initializeLog("starting NER, getting lines...")
   val lines = reformatEntityTags(getLines(args(0)))
   entityCounts.toSeq.sortBy(_._2)(Ordering[Integer].reverse) foreach {
     case(entityName,count) => {
       log("training entity recognizer: " + entityName + "(count: "+count+")")
       log("training model")
       val model = trainModel(linesToStream(lines), entityName)
       log("instantiating namefinder")
       val nameFinder = new NameFinderME(model)
       log("running namefinder on lines")
       val results = getResults(nameFinder,linesToStream(lines))
       log(results)
       if(S3_EXPORT) exportModel(model, entityName)
     }
   }
  }
  

  
  def exportModel(model:BaseModel, entityName:String) = {
    val modelFileName = entityName+".bin"
    val fos = new FileOutputStream(modelFileName);
    val oos = new ObjectOutputStream(fos);
    model.serialize(oos)
    fos.close
    oos.close
    if(S3_EXPORT) export(new java.io.File(modelFileName))
  }
  
  def initializeLog(s:String) = {
    val fw = new FileWriter(LOG_FILENAME)
    fw.write(s)
    fw.close()
  }
  
  def log(x:String) = {
    println(x)
    val fw = new FileWriter(new File(LOG_FILENAME), true)
    fw.write(x+"\n")
    fw.close
    if(S3_EXPORT) export(new File(LOG_FILENAME))
  }
  
  def export(x:java.io.File) = {
    val s3 = new AmazonS3Client(new BasicAWSCredentials(System.getenv("AWS_ACCESS_KEY_ID"), System.getenv("AWS_SECRET_ACCESS_KEY")))
    s3.putObject(BUCKET_NAME,REMOTE_AWS_DIR+"/"+x.getName, new File(x.getAbsolutePath()))
  }
  
  def linesToStream(lines:Array[String]) = {
    val charset = Charset.forName(CHARSET)
    new NameSampleDataStream(
        new PlainTextByLineStream(
            new ByteArrayInputStream(lines.mkString("\n").getBytes(CHARSET)), charset))
  }
  
  def reformatEntityTags(lines:Array[String]) = {
    val entityRegex = "\\[[A-Z]+\\s.+\\]\\s*"
    // for each line, if a span matches the definition of an entity, reformat only that span,
    // put back into the line, return the line
    lines map { line =>
      if(line.matches(".*"+entityRegex+".*")){
        val sb = new StringBuilder
        // do zero-width splits before each entity
        //guarantees that each split contains nothing or starts with an entity
        val entitySplits = line.split("(?=("+entityRegex+"))")
        for( i <- 0 until entitySplits.size)  { 
          val splitMaybeContainingEntity = entitySplits(i)
          if(splitMaybeContainingEntity.matches(".*"+entityRegex+".*")) {
            var endIndexOfEntity = 0
            while(!splitMaybeContainingEntity.slice(0,endIndexOfEntity).matches(entityRegex)) endIndexOfEntity += 1
            // append the reformatted entity and the rest of the string
            sb append reformatEntityTag(splitMaybeContainingEntity.slice(0, endIndexOfEntity)) + 
            splitMaybeContainingEntity.slice(endIndexOfEntity,splitMaybeContainingEntity.size)
          } else sb append splitMaybeContainingEntity // if split doesn't contain an entity, just put it back
        }
        sb.toString
      } else line // just map to the original line if there's no entity in it
    }
  }
  
  /**
   * Takes in an entity tag of some format, reformats it to OpenNLP format
   * Implementation function will have to be supplied out for each new data set
   * @args: entityTag, a String representing
   */
  def reformatEntityTag(entityTag:String):String = {
    DATA_SET match {
      case "REFLEX_ENG" => {reformatReflexEngEntityTag(entityTag)}
    }
  }
  
  def reformatReflexEngEntityTag(entityTag:String) = {
    var startTag = false
    val entityTagStringBuilder = new StringBuilder()
    val entityStringBuilder = new StringBuilder()
    
    val entity = ""
    entityTag foreach { char =>
      char match
      {
        case '[' => { entityTagStringBuilder append "<START:"; startTag = true }
        case ']' => entityTagStringBuilder append "<END>"
        case ' ' => { if(startTag){ entityTagStringBuilder append "> "; startTag = false } else entityTagStringBuilder append " " }
        case _ => { if(startTag) entityStringBuilder append char; entityTagStringBuilder append char }
      }
    }
    val entityString = entityStringBuilder.toString
    if(!entityCounts.contains(entityString)) entityCounts(entityString) = 0
    entityCounts(entityString) = entityCounts(entityString) +1
    entityTagStringBuilder.toString
  }
  
  def getLines(dir:String) = {
      var lines = MutableList[String]()
      val filesDir = new File(dir)
      val files = filesDir.listFiles.filterNot(_.isDirectory)
      files map { file =>
        val scanner = new Scanner(file)
        while(scanner.hasNextLine) lines += scanner.nextLine
      }
      lines.toArray
  }
  
  def trainModel(sampleStream:NameSampleDataStream, entityName:String) = {
      // the byte[] is a feature generator... supply later
      NameFinderME.train("en", entityName, sampleStream, TrainingParameters.defaultParams(),
            null:Array[Byte], Collections.emptyMap[String, Object]());
  }
  
  def getResults(nameFinder:TokenNameFinder, sampleStream:NameSampleDataStream) = {
    val listener = new TokenNameFinderDetailedFMeasureListener()
    val evaluator = new TokenNameFinderEvaluator(nameFinder, listener)
    evaluator.evaluate(sampleStream)
    listener.createReport
  }
}
