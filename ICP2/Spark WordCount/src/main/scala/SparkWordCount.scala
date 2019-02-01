

import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by Mayanka on 09-Sep-15.
  */
object SparkWordCount {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir", "C:\\Users\\lakshmikorrapati\\Downloads\\hadoop 2.8.1");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.textFile("input")

    val wc=input.flatMap(line=>{line.split(" ")}).groupBy(word=>(word.charAt(0))).cache()

    val output=wc

    output.saveAsTextFile("output")

    val o=output.collect()

    var s:String="Words:Count \n"
    o.foreach{case(word,count)=>{

      s+=word+" : "+count+"\n"


    }
    }

  }

}
