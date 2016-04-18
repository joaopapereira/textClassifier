package uk.co.jpereira;
import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.Classifier;
import uk.co.jpereira.bayes.BayesClassifier;
import com.da

import java.util.Arrays;
import java.util.Collection;

/**
 * Created by joaopereira on 4/16/2016.
 */
public class Application {


    public static void main(String[] args) {
        /*
             * Create a new classifier instance. The context features are
             * Strings and the context will be classified with a String according
             * to the featureset of the context.
             */
        final Classifier<String, String> bayesP =
                new BayesClassifier<String, String>(true);
        final Classifier<String, String> bayesA =
                new BayesClassifier<String, String>(false);

        for(String features: Data.cook)
            bayesP.learn("cook", Arrays.asList(features));

        for(String features: Data.softwareDeveloper)
            bayesP.learn("software developer", Arrays.asList(features));

        System.out.println( // will output "cook"
                bayesP.classify(Arrays.asList(Data.cook[0])).getCategory());
        System.out.println( // will output "software developer"
                bayesP.classify(Arrays.asList(Data.sdTest[0])).getCategory());


            /*
             * The BayesClassifier extends the abstract Classifier and provides
             * detailed classification results that can be retrieved by calling
             * the classifyDetailed Method.
             *
             * The classification with the highest probability is the resulting
             * classification. The returned List will look like this.
             * [
             *   Classification [
             *     category=negative,
             *     probability=0.0078125,
             *     featureset=[today, is, a, sunny, day]
             *   ],
             *   Classification [
             *     category=positive,
             *     probability=0.0234375,
             *     featureset=[today, is, a, sunny, day]
             *   ]
             * ]
             */
        Collection<Classification<String, String>> res = ((BayesClassifier<String, String>) bayesP).classifyDetailed(
                Arrays.asList(Data.sdTest[0]));
        System.out.println(res.iterator().next().getProbability());
        System.out.println(res.iterator().next().getProbability());
        System.out.println(bayesA);
            /*
             * Please note, that this particular classifier implementation will
             * "forget" learned classifications after a few learning sessions. The
             * number of learning sessions it will record can be set as follows:
             */
        //bayes.setMemoryCapacity(500); // remember the last 500 learned classifications
    }
}
