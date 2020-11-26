import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import java.util.*;
import java.util.Random;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

public class G21HW2 {

    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Favaro Simone
    // Molon Alberto
    // GROUP 21
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static void main(String[] args) throws Exception {

        // make sure that two parameters are passed
        if (args.length != 2)
            throw new IllegalArgumentException("Expecting two parameters (file name and k)");

        String filename = args[0];			                    // reading the input points into inputPoints ArrayList
        ArrayList<Vector> inputPoints = new ArrayList<>();      // set S
        inputPoints = readVectorsSeq(filename);

        int k = Integer.valueOf(args[1]);	                    // reading k value from args[1]
        // make sure that the k value is less than the size of inputPoints ArrayList
        if(k > inputPoints.size())
            throw new IllegalArgumentException("Expecting k less then the size of the input dataset");

        // initialization of variables needed for measuring the running times of each method
        long start, end;
        // initialization of a variable needed for measuring the max distance returned by each method
        double maxDist = 0.0;


        // running exactMPD(inputPoints), measuring its running time
        start = System.currentTimeMillis();
        maxDist = exactMPD(inputPoints);
        end = System.currentTimeMillis();
        System.out.println("\nEXACT ALGORITHM");
        System.out.println("Max distance = " + maxDist);
        System.out.println("Running time = " + (end - start) + " ms");

        // running twoApproxMPD(inputPoints, k), measuring its running time
        start = System.currentTimeMillis();
        maxDist = twoApproxMPD(inputPoints, k);
        end = System.currentTimeMillis();
        System.out.println("\n2-APPROXIMATION ALGORITHM");
        System.out.println("k = " + k);
        System.out.println("Max distance = " + maxDist);
        System.out.println("Running time = " + (end - start) + " ms");

        // running kcenterMPD(inputPoints, k), storing its result in centers ArrayList, running exactMPD(centers) and measuring the combined running time
        start = System.currentTimeMillis();
        ArrayList<Vector> centers = kcenterMPD(inputPoints, k);
        maxDist = exactMPD(centers);
        end = System.currentTimeMillis();
        System.out.println("\nk-CENTER-BASED ALGORITHM");
        System.out.println("k = " + k);
        System.out.println("Max distance = " + maxDist);
        System.out.println("Running time = " + (end - start) + " ms");
    }

    // auxiliary methods used for reading the input
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    //exactMPD method
    public static double exactMPD(ArrayList<Vector> inputPoints){       // it receives a Vector ArrayList as input
        int numb = inputPoints.size();                      // numb is the size of the input ArrayList
        double dist = 0.0;                                  // init of a temp variable for computing distance
        double maxDist = 0.0;                               // init of the output

        for(int i=0; i<numb; i++)   // first for loop cicles all values of inputPoints
        {
            for(int j=i+1; j<numb; j++) // second for loop cicles only Vectors in inputPoint greater than the i-th one, in order to avoid unnecessary calculations of symmetric distances
            {
                dist = Math.sqrt(Vectors.sqdist(inputPoints.get(i),inputPoints.get(j)));    // distance from the i-th to the j-th Vectors of inputPoints, using Vectors.sqdist() method
                if(dist > maxDist)
                    maxDist = dist;     // saving the greater distance each cicle
            }
        }
        return maxDist;         // it is the max distance returned as output
    }

    //twoApproxMPD method
    public static double twoApproxMPD(ArrayList<Vector> inputPoints, int k){    // it receives a Vector ArrayList and the int k as input
        int numb = inputPoints.size();                   // number of points
        double dist = 0.0;                               // init of a temp variable for computing distance
        double maxDist = 0.0;                            // init of the output

        final long SEED = 1237327;                      // init the seed constant as the ID badge number
        Random rnd = new Random();                      // init the random generator rnd
        rnd.setSeed(SEED);                              // init the seed for our random generator

        ArrayList<Vector> rndPoints = new ArrayList<>();    // set S'

        // fill the set S' with k points choose randomly from the set S
        for(int i=0; i<k; i++)
            rndPoints.add(i,inputPoints.get(rnd.nextInt(numb)));

        for(int i=0; i<k; i++)                  // first for loop cicles all values of rndPoints (set S')
        {
            for(int j=0; j<numb; j++)   // second for loop cicles all values of inputPoints (set S)
            {
                dist = Math.sqrt(Vectors.sqdist(rndPoints.get(i),inputPoints.get(j)));  // distance from the i-th rndPoints' Vector to the j-th inputPoints' Vector, using Vectors.sqdist() method
                if(dist > maxDist)
                    maxDist=dist;       // saving the greater distance each cicle
            }
        }
        return maxDist;                 // it is the max distance returned as output
    }

    //kcenterMPD method
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
                            index = j;
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
                            index = j;
                        }
                    }
                }
                centers.add(maxCenter);                     // add the the temp center to the list of centers (after the end of the second loop)
                //inputPoints.remove(index);              // remove the center from the set of point inputPoints
                //dist.remove(index);
                maxDist=0.0;                                // set to zero the max distance
            }
        return centers;                 // it is the set of centers returned as output
    }
}
