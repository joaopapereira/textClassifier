package uk.co.jpereira;
import de.daslaboratorium.machinelearning.classifier.Classification;
import de.daslaboratorium.machinelearning.classifier.Classifier;
import uk.co.jpereira.bayes.BayesClassifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

/**
 * Created by joaopereira on 4/16/2016.
 */
public class Application {
    public static Collection<String> prepareText(String text) {
        Collection<String> textAux = new ArrayList<>();
        String currentLine;
        for(String line: text.split("\\s")) {
            currentLine = line.replaceAll("\\b(up|to|all|with|by|other|a|or|and|then|must|least|i|am|of|but|our|mine|very|worked|decided|each|an|as|at|on|in)\\b", "")
                    .replaceAll("[^a-zA-Z\\d]", "");
            textAux.add(currentLine);
        }
        return textAux;
    }

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


        for(String features: Data1.administrator_assistant)
            bayesP.learn("administrative assistant", prepareText(features));
        for(String features: Data1.line_cook)
            bayesP.learn("cook", prepareText(features));
        for(String features: Data1.sales_manager)
            bayesP.learn("sales manager", prepareText(features));

        for(String features: Data1.secretary)
            bayesP.learn("secretary", prepareText(features));
        for(String features: Data1.software_developer)
            bayesP.learn("software developer", prepareText(features));
        boolean first = true;
        for(String features: Data.cook){
            if(first) {
                first = false;
                continue;
            }
            bayesP.learn("cook", prepareText(features));
        }


        for(String features: Data.softwareDeveloper)
            bayesP.learn("software developer", prepareText(features));

        System.out.println( // will output "cook"
                bayesP.classify(prepareText(Data.cook[0])).getCategory());
        System.out.println( // will output "software developer"
                bayesP.classify(prepareText(Data.sdTest[0])).getCategory());


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
                prepareText(Data.sdTest[0]));
        System.out.println(res.iterator().next().getProbability());
        System.out.println(res.iterator().next().getProbability());
        System.out.println(bayesA);
        res = ((BayesClassifier<String, String>) bayesP).classifyDetailed(
                prepareText(Data.cookTest[0]));
        System.out.println(res.iterator().next().getProbability());
            /*
             * Please note, that this particular classifier implementation will
             * "forget" learned classifications after a few learning sessions. The
             * number of learning sessions it will record can be set as follows:
             */
        //bayes.setMemoryCapacity(500); // remember the last 500 learned classifications
    }
}
