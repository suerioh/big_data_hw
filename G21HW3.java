import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import java.util.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


public class G21HW3 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Favaro Simone
    // Molon Alberto
    // GROUP 21
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {
        long start, end;	// variables for initialization time estimation

        start = System.currentTimeMillis();

        // Spark configuration
        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);		// object representing the master process
        sc.setLogLevel("WARN");

        // make sure that three parameters are passed
        if (args.length != 3)
            throw new IllegalArgumentException("Expecting three parameters (path file, k and L)");

        String filename = args[0];
        int k = Integer.parseInt(args[1]);	// reading k value from args[1]
        int L = Integer.parseInt(args[2]);	// reading L value from args[2]

        // creation of the JavaRDD (of points) from the file passed in input
        JavaRDD<Vector> inputPoints = sc.textFile(filename).map(G21HW3::strToVector).repartition(L).cache();

        long inputPointSize = inputPoints.count();
        if(k > inputPointSize/L || L > inputPointSize)
            throw new IllegalArgumentException("Expecting k and L less then the size of the input dataset");

        end = System.currentTimeMillis();

        System.out.println("Number of points = " + inputPointSize);
        System.out.println("k = " + k);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (end - start) + " ms");

        ArrayList<Vector> solution = runMapReduce(inputPoints, k , L);

        System.out.println("\n" + "Average distance = " + measure(solution));
    }	// END main


    // auxiliary method used for reading the input and returning the Vector object
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }	// END strToVector


    // runSequential method --> Sequential 2-approximation based on matching
    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    }	// END runSequential


    // kcenterMPD method
    public static ArrayList<Vector> kcenterMPD(ArrayList<Vector> inputPoints, int k){
        int index = 0;
        ArrayList<Double> dist = new ArrayList<>();         // init of a temp variable for computing distance
        double maxDist = 0.0;                               // init of the max distance
        Vector maxCenter = Vectors.zeros(1);             // init of a temp variable for computing centers
        ArrayList<Vector> centers = new ArrayList<>();      // init of the output

        centers.add(0,inputPoints.get(0));          // c1 is the first center, which is arbitrary (the first point of set S)
        for(int i=0; i<k-1; i++)           // first loop cicles for k-2 times because first center is already determine
        {
            for(int j=0; j<inputPoints.size(); j++)       // second loop cicles for all points of inputPoints (set S)
            {
                if(i == 0)
                {
                    dist.add(Math.sqrt(Vectors.sqdist(centers.get(i),inputPoints.get(j))));
                    if(dist.get(j) > maxDist)
                    {
                        maxCenter = inputPoints.get(j);     // saving the point which has the max distance (so it is a possible center)
                        maxDist = dist.get(j);
                    }
                }
                else    // only for i >= 1
                {
                    if(dist.get(j) > Math.sqrt(Vectors.sqdist(centers.get(i),inputPoints.get(j))))   //find the closest center from every point
                        dist.set(j,Math.sqrt(Vectors.sqdist(centers.get(i),inputPoints.get(j))));
                    if(dist.get(j) > maxDist)
                    {
                        maxCenter = inputPoints.get(j);     // saving the point which has the max distance (so it is a possible center)
                        maxDist = dist.get(j);
                    }
                }
            }
            centers.add(maxCenter);                     // add the the temp center to the list of centers (after the end of the second loop)
            maxDist=0.0;                                // set to zero the max distance
        }
        return centers;                 // it is the set of centers returned as output
    }	// END kcenterMPD


    // method that implements the 4-approximation MapReduce algorithm for diversity maximization
    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD, int k, int L){
        long start, end;	// variables for time estimation
        start = System.currentTimeMillis();
        JavaRDD<Vector> tempcoreset;

        List<Vector> coreset;

        tempcoreset = pointsRDD
                .mapPartitions((pointVector) -> {			// Reduce phase of ROUND 1
                    ArrayList<Vector> temp = new ArrayList<>();
                    while (pointVector.hasNext()) {
                        Vector point = pointVector.next();
                        temp.add(point);
                    }
                    return kcenterMPD(temp, k).iterator();
                })
                .cache();		// store in cache (useful for time estimation)
        tempcoreset.count();
        end = System.currentTimeMillis();
        System.out.println("\n" + "Runtime of Round 1 = " + (end - start) + " ms");

        start = System.currentTimeMillis();
        // collect all the selected k points into the arrayList sol
        coreset = tempcoreset.collect();										// ROUND 2
        ArrayList<Vector> sol = runSequential(new ArrayList<>(coreset), k);
        end = System.currentTimeMillis();
        System.out.println("Runtime of Round 2 = " + (end - start) + " ms");
        return sol;

    }	// END runMapReduce


    // method that determines the average distance among given points passed as parameters
    public static double measure(ArrayList<Vector> pointSet){
        double sum = 0.0;
        double len = pointSet.size();

        // first for loop cicles all values of pointSet
        for(int i=0; i<len; i++)
        {
            // second for loop cicles only Vectors in pointSet greater than the i-th one in order to avoid unnecessary calculations of symmetric distances
            for(int j=i+1; j<len; j++)
            {
                // distance from the i-th to the j-th Vectors of pointSet, using Vectors.sqdist() method
                sum = sum + Math.sqrt(Vectors.sqdist(pointSet.get(i), pointSet.get(j)));
            }
        }

        // return the average distance
        return sum/((len*len-1)/2);
    }	// END measure
}