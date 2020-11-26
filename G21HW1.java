import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;





public class G21HW1 {

  public static void main(String[] args) throws IOException {
    // CHECKING NUMBER OF CMD LINE PARAMETERS
    // Parameters are: number_partitions, <path to file>
    if (args.length != 2) {
      throw new IllegalArgumentException("USAGE: num_partitions file_path");
    }
    // SPARK SETUP
    SparkConf conf = new SparkConf(true).setAppName("Homework1");
    JavaSparkContext sc = new JavaSparkContext(conf);
    sc.setLogLevel("WARN");
    // INPUT READING
    // Read number of partitions
    int K = Integer.parseInt(args[0]);
    // Read input file and subdivide it into K random partitions
    JavaRDD<String> lines = sc.textFile(args[1]).repartition(K);
    // SETTING GLOBAL VARIABLES
    JavaPairRDD<String, Long> count;
    int N_max = 0;
    for(int k = 0; k < 4; k++){
        int temp = lines.glom().collect().get(k).size();
        if(temp >= N_max){
            N_max = temp;
        }
    }





    // Version with deterministic partitions
    // the code is essentialy the same of template's second part: we have added comments just to the few lines of code we had to edit
    count = lines
            .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)
                String[] tokens = line.split(" ");
                // need to convert the string to an integer
                int token_int = Integer.parseInt(tokens[0]);
                ArrayList<Tuple2<Integer, String>> pairs = new ArrayList<>();
                // creating the pair using the way described in the assignment rules
                pairs.add(new Tuple2<>(token_int % K, tokens[1]));
                return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R1)
            .flatMapToPair((pair) -> {
                HashMap<String, Long> counts = new HashMap<>();
                // c takes the value of each iterator's string (that is the value field of the pair)
                for (String c : pair._2()) {
                    counts.put(c, 1L + counts.getOrDefault(c, 0L));
                }
                ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                for (Map.Entry<String, Long> e : counts.entrySet()) {
                    pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                }
                return pairs.iterator();
            })
            .groupByKey()    // <-- REDUCE PHASE (R2)
            .mapValues((it) -> {
                long sum = 0;
                for (long c : it) {
                    sum += c;
                }
                return sum;
            });
    // print results
    System.out.println("VERSION WITH DETERMINISTIC PARTITIONS");
    System.out.print("Output pairs = ");
    for(Tuple2<String,Long> tuple:count.sortByKey().collect()) {
        System.out.print("(" + tuple._1() + "," + tuple._2() + ") ");
    }





    // Version with Spark partitions
    count = lines
            .flatMapToPair((line) -> {    // <-- MAP PHASE (R1)
                String[] tokens = line.split(" ");
                // need to convert the string to an integer
                int token_int = Integer.parseInt(tokens[0]);
                ArrayList<Tuple2<Integer, String>> pairs = new ArrayList<>();
                // creating the pair using the way described in the assignment rules
                pairs.add(new Tuple2<>(token_int, tokens[1]));
                return pairs.iterator();
            })
            .mapPartitionsToPair((cc) -> {    // <-- REDUCE PHASE (R1)
                 HashMap<String, Long> counts = new HashMap<>();
                 while (cc.hasNext()){
                      Tuple2<Integer, String> tuple = cc.next();
                      counts.put(tuple._2(), 1L + counts.getOrDefault(tuple._2(), 0L));
                 }
                 ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                 for (Map.Entry<String, Long> e : counts.entrySet()) {
                     pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                 }
                 return pairs.iterator();
            })
            .groupByKey()     // <-- REDUCE PHASE (R2)
            .mapValues((it) -> {
                 long sum = 0;
                 for (long c : it) {
                     sum += c;
                 }
                 return sum;
            });
    System.out.println("");
    System.out.println("VERSION WITH SPARK PARTITIONS");
    System.out.print("Most frequent class = ");
    Tuple2<String,Long> tuple_maxvalue = new Tuple2<>("", 0L);
    for(Tuple2<String,Long> tuple : count.collect()) {
        if (tuple._2() >= tuple_maxvalue._2()) {
            tuple_maxvalue = tuple;
        }
    }
    System.out.println("(" + tuple_maxvalue._1() + "," + tuple_maxvalue._2() + ") ");
    System.out.print("Most frequent class = " + N_max);

  }
}