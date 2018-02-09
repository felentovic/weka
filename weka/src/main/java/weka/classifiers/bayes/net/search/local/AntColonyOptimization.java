package weka.classifiers.bayes.net.search.local;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.core.*;

import java.util.*;
import java.util.concurrent.*;

/**
 * Created by Borna Feldsar on 26/08/17.
 * This structure search algorithm is based on the ant colony optimization. It is an implementation of the paper
 * Ant colony optimization for learning Bayesian networks, Luis M. de Campos , Juan M. Fernandez-Luna,Jose A.Gamez, Jose M.Puerta
 * This implementation is adopted to the Weka environment and allows parameter settings through GUI. Ants solution building and
 * local optimizer solution optimization allows parallelization with the number of threads equal to a number of available cores.
 * In order to recreate the same solution when the same seed value is given, local pheromone update is done after an ant builds
 * a solution and not while building it. More local search algorithms are available as local optimizers.
 */
public class AntColonyOptimization extends LocalScoreSearchAlgorithm {

    /**
     * cache for remembering the change in score for steps in the search space
     */
    class Cache implements RevisionHandler {

        /**
         * change in score due to adding an arc
         **/
        double[][] m_fDeltaScoreAdd;

        /**
         * product of a pheromone value and a heuristic value powered by beta for an arc
         */

        double[][] m_arcPheromoneHeuristicB;
        /**
         * number of arcs which are still candidates ( not in G and inclusion of an arc improves G)
         */
        int numOfAvailableArcs;
        /**
         * sum of cells that are still candidates  ( not in G and inclusion of an arc improves G)
         **/
        double sumOfPheromoneHeuristic;

        /**
         * constructor
         *
         * @param nNrOfNodes number of nodes in a network, used to determine a memory
         *                   size to reserve
         */
        Cache(int nNrOfNodes) {
            m_fDeltaScoreAdd = new double[nNrOfNodes][nNrOfNodes];
            m_arcPheromoneHeuristicB = new double[nNrOfNodes][nNrOfNodes];

        }

        /**
         * Set cache entry
         *
         * @param m_nHead index of a head node
         * @param m_nTail index of a tail node
         * @param fValue  value to put in the cache
         */
        public void putScore(int m_nTail, int m_nHead, double fValue) {
            m_fDeltaScoreAdd[m_nTail][m_nHead] = fValue;
        } // put

        /**
         * Get cache entry
         *
         * @return cache value
         */
        public double getScore(int m_nTail, int m_nHead) {
            return m_fDeltaScoreAdd[m_nTail][m_nHead];

        } // get

        /**
         * Set cache entry
         *
         * @param m_nHead index of a head node
         * @param m_nTail index of a tail node
         * @param fValue  value to put in the cache
         */
        public void putPheromoneHeuristicB(int m_nTail, int m_nHead, double fValue) {
            m_arcPheromoneHeuristicB[m_nTail][m_nHead] = fValue;
        } // put

        /**
         * Get cache entry
         *
         * @return cache value
         */
        public double getPheromoneHeuristicB(int m_nTail, int m_nHead) {
            return m_arcPheromoneHeuristicB[m_nTail][m_nHead];
        } // get

        /**
         * Returns the revision string.
         *
         * @return the revision
         */
        @Override
        public String getRevision() {
            return RevisionUtils.extract("$Revision$");
        }
    } // class Cache

    private class Ant extends LocalScoreSearchAlgorithm implements Callable<Ant> {
        /**
         * cache
         **/
        Cache m_Cache = null;

        /**
         * exponent of a diff score cell in the formula (11)
         **/
        double f_beta;
        /**
         * selection probability in the formula (10)
         **/
        double q0;
        /**
         * initial pheromone level
         **/
        double f_pheromone0;
        /**
         * coefficient in the local pheromone update
         **/
        double f_localUpdateCoef;
        /**
         * true if there is an arc (tail,head)
         */
        boolean[][] m_arcs;

        private Random randomNumberGenerator = new Random();

        private Instances instances;

        @Override
        public void search(BayesNet bayesNet, Instances instances) throws Exception {
            m_arcs = new boolean[instances.numAttributes()][instances.numAttributes()];
            initCache(instances.numAttributes());

            while (m_Cache.numOfAvailableArcs > 0) {
                int[] indices = selectIndices(instances.numAttributes(), q0);
                int attributeTail = indices[0];
                int attributeHead = indices[1];

                if (attributeTail == -1 || attributeHead == -1) {
                    //should not happen because of the condition above
                    break;
                }
                //add Tail in the parent set of Head
                ParentSet parentSet = bayesNet.getParentSet(attributeHead);
                parentSet.addParent(attributeTail, instances);
                // set value to used
                m_Cache.putScore(attributeTail, attributeHead, Double.NEGATIVE_INFINITY);
                m_arcs[attributeTail][attributeHead] = true;

                updateAncestorDescendantArcs(attributeTail, attributeHead, bayesNet, instances.numAttributes());
                updateCacheMatrices(attributeHead, instances.numAttributes());
                updateProporionalSelectionValues(instances.numAttributes());
            }
        }

        /**
         * Decrease the pheromone value of already used arcs in order to enable exploration of the search space
         *
         * @param nNrOfAtts number of attributes in the data set
         */
        public void performLocalPheromoneUpdate(int nNrOfAtts) {
            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    if (m_arcs[iAttributeTail][iAttributeHead]) {
                        localPheromoneUpdate(iAttributeTail, iAttributeHead);
                    }
                }
            }
        }

        /**
         * Preforms local pheromone update making an arc less desirable. Method is multithread safe
         *
         * @param attributeTail arc's tail
         * @param attributeHead arc's head
         */
        private void localPheromoneUpdate(int attributeTail, int attributeHead) {
            pheromone[attributeTail][attributeHead] = (1 - f_localUpdateCoef) * pheromone[attributeTail][attributeHead]
                    + f_localUpdateCoef * f_pheromone0;
        }

        /**
         * Select two indices (arc) from pair candidates. An arc that gives the most improvement is selected with the probability q0,
         * and with probability (1 - q0) an arc is selected regarding its probability, which is calculated using the formula (11)
         * in the paper.
         *
         * @param nNrOfAtts number of attributes in the data set
         * @param q0        probability of method for selecting arcs
         * @return an arc as an array. At the index 0 is Tail and at the index 1 is Head. Returns {-1,-1} if no arc addition is
         * allowed
         */
        private int[] selectIndices(int nNrOfAtts, double q0) {
            int[] arcs;
            if (randomNumberGenerator.nextDouble() < q0) {
                arcs = findBestArc(nNrOfAtts);
            } else {
                arcs = proportionallySelectIndices(nNrOfAtts);
            }
            return arcs;
        }

        /**
         * Find the best arc regarding the product of score diff and a pheromone value powered by f_beta coef
         *
         * @param nNrOfAtts number of attributes in the data set
         * @return an arc as an array. At the index 0 is Tail and at the index 1 is Head. Returns {-1,-1} if no arc addition is
         * allowed
         */
        private int[] findBestArc(int nNrOfAtts) {
            int[] bestArc = new int[]{-1, -1};
            double bestScore = -1;
            // find the best arc
            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    if (checkIfArcIsInFg(iAttributeTail, iAttributeHead)) {
                        double valPhHeuB = m_Cache.getPheromoneHeuristicB(iAttributeTail, iAttributeHead);
                        if (bestArc[0] == -1 || valPhHeuB > bestScore) {
                            bestArc[0] = iAttributeTail;
                            bestArc[1] = iAttributeHead;
                            bestScore = valPhHeuB;
                        }
                    }
                }
            }
            return bestArc;
        }//findBestArc

        /**
         * Selects arc proportionally using the probability calculated by the formula (11) in the paper
         *
         * @param nNrOfAtts number of attributes in the data set
         * @return an arc as an array. At the index 0 is Tail and at the index 1 is Head. Returns {-1,-1} if no arc addition is
         * allowed
         */
        private int[] proportionallySelectIndices(int nNrOfAtts) {
            int[] indices = new int[]{-1, -1};
            double randValue = randomNumberGenerator.nextDouble() * m_Cache.sumOfPheromoneHeuristic;

            double accumulatedSum = 0;
            for (int iAttributeHead = 0, index = 0; iAttributeHead < nNrOfAtts; iAttributeHead++, index++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    if (checkIfArcIsInFg(iAttributeTail, iAttributeHead)) {
                        double tmpPhHeu = m_Cache.getPheromoneHeuristicB(iAttributeTail, iAttributeHead);
                        accumulatedSum += tmpPhHeu;
                        if (Double.compare(randValue, accumulatedSum) <= 0) {
                            indices[0] = iAttributeTail;
                            indices[1] = iAttributeHead;
                            return indices;
                        }
                    }

                }
            }
            return indices;
        }//proportionallySelectIndices

        /**
         * Update the cache due to the change of the parent set of a node. Updates only values which are candidates
         * for the parent set
         *
         * @param iAttributeHead node that has its parent set changed
         * @param nNrOfAtts      number of attributes in a data set
         */
        private void updateCacheMatrices(int iAttributeHead, int nNrOfAtts) {
            // update score cache entries for arrows heading towards iAttributeHead
            double fBaseScore = calcNodeScore(iAttributeHead);
            for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                //if an arc is not forbidden
                if (Double.compare(m_Cache.getScore(iAttributeTail, iAttributeHead), Double.NEGATIVE_INFINITY) != 0) {
                    // add entries to cache for adding arcs
                    double valScore = calcScoreWithExtraParent(iAttributeHead, iAttributeTail)
                            - fBaseScore;
                    m_Cache.putScore(iAttributeTail, iAttributeHead, valScore);
                    double valPhHeuB = pheromone[iAttributeTail][iAttributeHead] * Math.pow(valScore, f_beta);
                    m_Cache.putPheromoneHeuristicB(iAttributeTail, iAttributeHead, valPhHeuB);
                }
            }
        } // updateCacheMatrices

        /**
         * Updates a number of available arcs and a sum of cells of candidate arcs. It is used for proportionally
         * selecting indices
         *
         * @param nNrOfAtts number of attributes in the data set
         */
        private void updateProporionalSelectionValues(int nNrOfAtts) {
            double tmpSum = 0;
            int tmpAvailableArcs = 0;
            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    if (checkIfArcIsInFg(iAttributeTail, iAttributeHead)) {
                        //sum all positive values
                        tmpSum += m_Cache.getPheromoneHeuristicB(iAttributeTail, iAttributeHead);
                        tmpAvailableArcs += 1;
                    }
                }
            }
            m_Cache.sumOfPheromoneHeuristic = tmpSum;
            m_Cache.numOfAvailableArcs = tmpAvailableArcs;
        }//updateProporionalSelectionValues

        /**
         * Returns true if an arc is in the set Fg. Set Fg is defined as a set of arcs which have score bigger than 0,
         * their inclusion doesn't create a cycle and there are not in the graph already.
         *
         * @param iAttributeTail tail of an arc
         * @param iAttributeHead head of an arc
         * @return
         */
        private boolean checkIfArcIsInFg(int iAttributeTail, int iAttributeHead) {
            // check if the current arc score is bigger than 0 and it is not -inf
            return m_Cache.getScore(iAttributeTail, iAttributeHead) > 0;

        }

        /**
         * Initialize the cache entries
         *
         * @param nNrOfAtts number of attributes in the data set
         */

        private void initCache(int nNrOfAtts) {
            double[] fBaseScores = new double[nNrOfAtts];

            m_Cache = new Cache(nNrOfAtts);
            for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
                // determine base scores
                fBaseScores[iAttribute] = calcNodeScore(iAttribute);
            }

            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    double valScore;
                    double valPhHeuB;
                    if (iAttributeHead == iAttributeTail) {
                        valScore = Double.NEGATIVE_INFINITY;
                        valPhHeuB = Double.NEGATIVE_INFINITY;
                    } else {
                        valScore = calcScoreWithExtraParent(iAttributeHead, iAttributeTail)
                                - fBaseScores[iAttributeHead];
                        valPhHeuB = pheromone[iAttributeTail][iAttributeHead] * Math.pow(valScore, f_beta);

                    }
                    m_Cache.putScore(iAttributeTail, iAttributeHead, valScore);
                    m_Cache.putPheromoneHeuristicB(iAttributeTail, iAttributeHead, valPhHeuB);
                }
            }
            updateProporionalSelectionValues(nNrOfAtts);

        }// initCache

        /**
         * Creates two sets X and Y and forbids arcs with a tail in X and a head in Y. X set consists of ancestors of the attributeTail and attributeTail itself.
         * Y set consists of descendants of the attributeHead and the attributeHead itself. With that ban we prevent introduction of cycles.
         *
         * @param attributeTail   tail of an arc
         * @param attributeHead   head of an arc
         * @param bayesNet        Bayes network to be learned
         * @param numOfAttributes number of attributes in the data set
         */
        private void updateAncestorDescendantArcs(int attributeTail, int attributeHead, final BayesNet bayesNet, int numOfAttributes) {
            int initialListSize = numOfAttributes / 2;

            //all ancestors of AttributeTail
            List<Integer> ancestors = BFS(attributeTail, initialListSize, new Function() {

                @Override
                public List<Integer> generateNextLevel(int root) {
                    List<Integer> nextLevel = new ArrayList<>();
                    ParentSet parentSet = bayesNet.getParentSet(root);
                    int[] parents = parentSet.getParents();
                    for (int i = 0; i < parentSet.getNrOfParents(); i++) {
                        nextLevel.add(parents[i]);
                    }
                    return nextLevel;
                }


            });
            //all descendants of AttributeHead
            List<Integer> descendants = BFS(attributeHead, initialListSize, new Function() {
                @Override
                public List<Integer> generateNextLevel(int root) {
                    List<Integer> nextLevel = new ArrayList<>();
                    boolean[] descedants = m_arcs[root];
                    for (int iHead = 0; iHead < descedants.length; iHead++) {
                        if (descedants[iHead]) {
                            nextLevel.add(iHead);
                        }
                    }
                    return nextLevel;
                }
            });

            for (Integer iHead : ancestors) {
                for (Integer iTail : descendants) {
                    m_Cache.putScore(iTail, iHead, Double.NEGATIVE_INFINITY);
                }
            }
            //free up memory
            ancestors = null;
            descendants = null;
        }//updateAncestorDescendantArcs

        /**
         * Breath first search which requires a function for generating the next level of nodes
         *
         * @param rootNode    start node
         * @param initialSize initial size of a list
         * @param function    the way in which a next level of nodes is generated
         * @return list of all visited nodes
         */
        private List<Integer> BFS(int rootNode, int initialSize, Function function) {
            List<Integer> visitedNodes = new ArrayList<>(initialSize);
            visitedNodes.add(rootNode);
            List<Integer> listPrevLevel = new ArrayList<>(initialSize);
            List<Integer> listCurrLevel = new ArrayList<>(initialSize);
            listPrevLevel.add(rootNode);

            while (!listPrevLevel.isEmpty()) {
                //for every node add all of its direct neighbours in the next level
                for (Integer iNode : listPrevLevel) {
                    List<Integer> nextLevel = function.generateNextLevel(iNode);
                    listCurrLevel.addAll(nextLevel);
                }
                visitedNodes.addAll(listCurrLevel);
                listPrevLevel.clear();
                listPrevLevel.addAll(listCurrLevel);
                listCurrLevel.clear();
            }
            return visitedNodes;
        }//BFS

        /**
         * Sets the max number of parents
         *
         * @param nMaxNrOfParents the max number of parents
         */
        public void setMaxNrOfParents(int nMaxNrOfParents) {
            m_nMaxNrOfParents = nMaxNrOfParents;
        }

        /**
         * Gets the max number of parents.
         *
         * @return the max number of parents
         */
        public int getMaxNrOfParents() {
            return m_nMaxNrOfParents;
        }

        /**
         * Sets whether to init as naive bayes
         *
         * @param bInitAsNaiveBayes whether to init as naive bayes
         */
        public void setInitAsNaiveBayes(boolean bInitAsNaiveBayes) {
            m_bInitAsNaiveBayes = bInitAsNaiveBayes;
        }

        /**
         * Gets whether to init as naive bayes
         *
         * @return whether to init as naive bayes
         */
        public boolean getInitAsNaiveBayes() {
            return m_bInitAsNaiveBayes;
        }

        /**
         * Set coefficient beta
         *
         * @param f_beta coef beta
         */
        public void setF_beta(double f_beta) {
            this.f_beta = f_beta;
        }

        /**
         * Set coefficient f_q0
         *
         * @param f_q0 coef f_q0
         */
        public void setQ0(double f_q0) {
            this.q0 = f_q0;
        }

        /**
         * Set pheromone initial value
         *
         * @param f_pheromone0 initial value
         */
        public void setF_pheromone0(double f_pheromone0) {
            this.f_pheromone0 = f_pheromone0;
        }

        /**
         * Sets local update coefficient
         *
         * @param f_localUpdateCoef local update coef
         */
        public void setF_localUpdateCoef(double f_localUpdateCoef) {
            this.f_localUpdateCoef = f_localUpdateCoef;
        }

        /**
         * Sets seed for random generator
         *
         * @param seed seed
         */
        public void setSeed(long seed) {
            randomNumberGenerator.setSeed(seed);
        }

        /**
         * Sets instances
         *
         * @param instances instances
         */
        public void setInstances(Instances instances) {
            this.instances = instances;
        }

        @Override
        public Ant call() throws Exception {
            BayesNet currentBayesNet = new BayesNet();
            currentBayesNet.m_Instances = instances;
            currentBayesNet.initStructure();
            buildStructure(currentBayesNet, instances);

            return this;
        }

    }//class Ant

    /**
     * Interface used in imitate recursion method
     */
    private interface Function {
        public List<Integer> generateNextLevel(int root);
    }

    /**
     * pheromone level matrix
     */
    private double[][] pheromone;
    /**
     * coefficient in global pheromone update
     */
    private double f_globalUpdateCoef = 0.4;
    /**
     * used for initial solution
     */
    private K2 k2;
    /**
     * number of iterations
     */
    private int numOfIterations = 50;
    /**
     * number of ants
     */
    private int numOfAnts = 10;
    /**
     * exponent of diff score cell in formula (11)
     **/
    private double f_beta = 2.0;
    /**
     * selection probability  formula (10)
     **/
    private double q0 = 0.8;
    /**
     * initial pheromone level
     **/
    private double f_pheromone0;
    /**
     * coefficient in local pheromone update
     **/
    private double f_localUpdateCoef = 0.4;
    /**
     * seed used for random in arc selection in ant
     */
    private long seed = 1;
    /**
     * after optimizationStep iterations perform local optimization
     */
    private int optimizationStep = 10;
    /**
     * selected local optimizer
     */
    private Optimizer selectedOptimizer = Optimizer.HCST;
    /**
     * parallelization flag
     */
    private boolean parallelize = true;

    @Override
    protected void search(BayesNet bayesNet, Instances instances) throws Exception {
        //no boundary to max nr of parents
        m_nMaxNrOfParents = 100000;
        //create initial solution with K2
        k2 = new K2();
        k2.setScoreType(getScoreType());
        k2.setMaxNrOfParents(m_nMaxNrOfParents);
        k2.buildStructure(bayesNet, instances);

        //calculate score of k2
        double totalK2Score = calculateNetworkScore(k2, instances.numAttributes());

        //init pheromone matrix
        f_pheromone0 = 1 / (instances.numAttributes() * Math.abs(totalK2Score));
        initPheromoneMatrix(instances.numAttributes(), f_pheromone0);

        // keeps track of the best structure found so far
        BayesNet bestBayesNet;

        // initialize bestBayesNet with K2 bayes net
        double fBestScore = totalK2Score;
        bestBayesNet = new BayesNet();
        bestBayesNet.m_Instances = instances;
        bestBayesNet.initStructure();
        copyParentSets(bestBayesNet, bayesNet);
        System.out.println("Initial sore:" + fBestScore);


        //create ExecutorService
        int numOfThreads = parallelize ? Runtime.getRuntime().availableProcessors() : 1;
        ExecutorService executor = Executors.newFixedThreadPool(numOfThreads);

        System.out.println("Optimizer " + selectedOptimizer + " selected.");

        //store all callable ants here
        List<Callable<Ant>> callablesAnts = new ArrayList<>();
        //store all results of threads executions
        List<Ant> antsResults = new ArrayList<>();

        //start iterations
        for (int iteration = 0; iteration < numOfIterations; iteration++) {
            callablesAnts.clear();
            //create ants and add them to the list
            for (int antNum = 0; antNum < numOfAnts; antNum++) {
                //create an ant
                Ant ant = new Ant();
                initializeAntParameters(ant, instances, antNum + iteration * 100);
                callablesAnts.add(ant);
            }

            antsResults.clear();
            List<Future<Ant>> futuresAnts = executor.invokeAll(callablesAnts);
            //collect results and store them
            for (Future<Ant> future : futuresAnts) {
                Ant antResult = null;
                try {
                    antResult = future.get();
                    antsResults.add(antResult);
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                    continue;
                }

            }

            if (iteration % optimizationStep == 0 || iteration + 1 == numOfIterations) {
                executeOptimizersParallel(antsResults, executor);
            }

            int bestAntindex = -1;
            for (int antNum = 0; antNum < numOfAnts; antNum++) {
                Ant currentAnt = antsResults.get(antNum);

                //do update in the same thread to recreate a solution with the same seed
                currentAnt.performLocalPheromoneUpdate(instances.numAttributes());

                double fAntScore = calculateNetworkScore(currentAnt, instances.numAttributes());

                if (fAntScore > fBestScore) {
                    fBestScore = fAntScore;
                    bestAntindex = antNum;
                }
            }

            // if better solution is found
            if (bestAntindex != -1) {
                copyParentSets(bestBayesNet, antsResults.get(bestAntindex).m_BayesNet);
            }

            if (iteration % optimizationStep == 0 || iteration + 1 == numOfIterations) {
                System.out.println("Best after iteration " + iteration + ".:" + fBestScore);
            }

            //global pheromone update
            globalPheromoneUpdate(bestBayesNet, fBestScore, instances.numAttributes());
        }


        // restore current network to the best network
        copyParentSets(bayesNet, bestBayesNet);

    }//search

    /**
     * Executes local optimization of ants solutions in parallel.
     *
     * @param antsResults list of ants with built solution
     * @param executor    executor service
     */
    private void executeOptimizersParallel(List<Ant> antsResults, ExecutorService executor) {
        List<Future> futuresOptimizers = new ArrayList<>();

        for (int antNum = 0; antNum < numOfAnts; antNum++) {
            //create an optimizer
            HillClimber localOptimizer = getLocalOptimizer();
            initLocalOptimizer(localOptimizer);
            //set the same bayes net object
            localOptimizer.m_BayesNet = antsResults.get(antNum).m_BayesNet;
            futuresOptimizers.add(executor.submit(localOptimizer));
        }

        for (Future optimizer : futuresOptimizers) {
            try {
                optimizer.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

    }

    /**
     * Returns network score which is the sum of scores of individual nodes
     *
     * @param algorithm  algorithm which has the bayesian net structure
     * @param numOfAttrs number of attributes in the data set
     * @return network score
     */
    private double calculateNetworkScore(LocalScoreSearchAlgorithm algorithm, int numOfAttrs) {
        double totalScore = 0;
        for (int iAttribute = 0; iAttribute < numOfAttrs; iAttribute++) {
            totalScore += algorithm.calcNodeScore(iAttribute);
        }
        return totalScore;
    }

    /**
     * Initialize pheromone matrix by allocating the memory and setting initial values to every cell
     *
     * @param numOfAttrs   number of attributes in the data set
     * @param f_pheromone0 initial pheromone value
     */
    private void initPheromoneMatrix(int numOfAttrs, double f_pheromone0) {
        pheromone = new double[numOfAttrs][numOfAttrs];
        for (int i = 0; i < numOfAttrs; i++) {
            for (int j = 0; j < numOfAttrs; j++) {
                pheromone[i][j] = f_pheromone0;
            }
        }
    }

    /**
     * Performs global pheromone update by updating the pheromone value of arcs which are present in the best solution
     * found
     *
     * @param bestBayesNet best bayes structure found
     * @param fBestScore   score of the best structure
     * @param numOfAttrs   number of attributes in the data set
     */
    private void globalPheromoneUpdate(BayesNet bestBayesNet, double fBestScore, int numOfAttrs) {
        double reciprocalScore = 1 / Math.abs(fBestScore);
        for (int iAttributeHead = 0; iAttributeHead < numOfAttrs; iAttributeHead++) {
            ParentSet parentSet = bestBayesNet.getParentSet(iAttributeHead);
            for (int iAttributeTailIndex = 0; iAttributeTailIndex < parentSet.getNrOfParents(); iAttributeTailIndex++) {
                int iAttributeTail = parentSet.getParent(iAttributeTailIndex);
                pheromone[iAttributeTail][iAttributeHead] = (1 - f_globalUpdateCoef) * pheromone[iAttributeTail][iAttributeHead]
                        + f_globalUpdateCoef * reciprocalScore;
            }
        }
    }

    /**
     * Returns the new object of selected local optimizer
     *
     * @return new selected local optimzer
     */
    private HillClimber getLocalOptimizer() {
        switch (selectedOptimizer) {
            case HCST:
                return new HillClimber();

            case LAGDHC:
                return new LAGDHillClimber();

            case TABU:
                return new TabuSearch();
        }
        return null;
    }


    /**
     * Initializes values of local optimizer. All optimizers extend HillClimber
     *
     * @param hc local optimizer
     */
    private void initLocalOptimizer(HillClimber hc) {
        hc.setInitAsNaiveBayes(false);
        hc.setMaxNrOfParents(m_nMaxNrOfParents);
        hc.setScoreType(getScoreType());
        hc.setUseArcReversal(true);
    }

    /**
     * Initializes parameters for the given ant to match the selected one.
     *
     * @param ant        ant which parameters are set
     * @param instances  instances in a data set
     * @param seedOffset number which ensures that each ant has different seed. Introduce deterministic randomness
     */
    private void initializeAntParameters(Ant ant, Instances instances, int seedOffset) {
        //pheromone matrix and stuff
        ant.setMaxNrOfParents(m_nMaxNrOfParents);
        ant.setInitAsNaiveBayes(false);
        ant.setF_localUpdateCoef(f_localUpdateCoef);
        ant.setF_beta(f_beta);
        ant.setF_pheromone0(f_pheromone0);
        ant.setQ0(q0);
        ant.setSeed(seed + seedOffset);
        ant.setScoreType(getScoreType());
        ant.setInstances(instances);
    }

    /**
     * Copies parent sets of source to dest BayesNet
     *
     * @param dest   destination network
     * @param source source network
     */
    private void copyParentSets(BayesNet dest, BayesNet source) {
        int nNodes = source.getNrOfNodes();
        // clear parent set first
        for (int iNode = 0; iNode < nNodes; iNode++) {
            dest.getParentSet(iNode).copy(source.getParentSet(iNode));
        }
    } // CopyParentSets

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> newVector = new Vector<Option>(4);

        newVector.addElement(new Option("\tBeta coefficient", "B", 1,
                "-B <beta coefficient>"));
        newVector.addElement(new Option("\tQ0 coefficient", "Q", 1,
                "-Q <Q0 coefficient>"));
        newVector.addElement(new Option("\tExploration coefficient", "X", 1,
                "-X <exploration coefficient>"));
        newVector.addElement(new Option("\tEvaporation coefficient", "V", 1,
                "-V <evaporation coefficient>"));
        newVector.addElement(new Option("\tNumber of iterations of ACO", "I", 1,
                "-I <num of iterations>"));
        newVector.addElement(new Option("\tNumber of ants", "M", 1,
                "-M <num of ants>"));
        newVector.addElement(new Option("\tSeed", "S", 1,
                "-S <seed>"));
        newVector.addElement(new Option("\tLocal optimization step", "T", 1,
                "-H <iterationStep>"));
        newVector.addElement(new Option("\tLocal Optimizer (Hill Climber (A,D,R), LAGD Hill Climber, Tabu Search )",
                "opt", 1, "-opt [HCST|LAGDHC|TABU]"));
        newVector.addElement(new Option("\tParallelize.\n\t(default true)", "P",
                0, "-P"));
        newVector.addAll(Collections.list(super.listOptions()));


        return newVector.elements();
    } // listOptions

    /**
     * Parses a given list of options.
     * <p/>
     * <p>
     * <!-- options-start --> Valid options are:
     * <p/>
     * <pre>
     * -P &lt;nr of parents&gt;
     *  Maximum number of parents
     * </pre>
     * <p>
     * <pre>
     * -A
     *  alpha coefficient
     * </pre>
     * <p>
     * <pre>
     * -B
     *  Beta coefficient
     * </pre>
     * <p>
     * <pre>
     * -Q
     * Q0 coefficient
     * </pre>
     * <p>
     * <pre>
     *  -X
     *  Exploration coefficient
     * </pre>
     * <p>
     * <pre>
     *  -V
     *  Evaporation coefficient
     * </pre>
     * <p>
     * <pre>
     *  -I
     *  Number of iterations
     * </pre>
     * <p>
     * <pre>
     *  -M
     *  Number of ants
     * </pre>
     * <p>
     * <pre>
     *  -S
     *  Seed
     * </pre>
     * <p>
     * <pre>
     * -T
     * Local optimization step
     * </pre>
     * <p>
     * <pre>
     * -opt
     * Local optimizer
     * </pre>
     * <p>
     * <pre>
     * -O
     * Parallelize
     * </pre>
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {

        setF_beta(parseOptionDouble(Utils.getOption('B', options), f_beta));
        setQ0(parseOptionDouble(Utils.getOption('Q', options), q0));
        setF_localUpdateCoef(parseOptionDouble(Utils.getOption('X', options), f_localUpdateCoef));
        setF_globalUpdateCoef(parseOptionDouble(Utils.getOption('V', options), f_globalUpdateCoef));
        setNumOfIterations(parseOptionInteger(Utils.getOption('I', options), numOfIterations));
        setNumOfAnts(parseOptionInteger(Utils.getOption('M', options), numOfAnts));
        setSeed(parseOptionLong(Utils.getOption('M', options), seed));
        setOptimizationStep(parseOptionInteger(Utils.getOption('T', options), optimizationStep));
        setLocalOptimizer(Utils.getOption("opt", options));
        setParallelize(Utils.getFlag('P', options));

        super.setOptions(options);
    } // setOptions

    /**
     * Parse given string and determine an optimizer by it
     *
     * @param optimizerString string to parse
     */
    private void setLocalOptimizer(String optimizerString) {
        if (optimizerString.equals(Optimizer.HCST.toString())) {
            setSelectedOptimizer(Optimizer.HCST);
        } else if (optimizerString.equals(Optimizer.LAGDHC.toString())) {
            setSelectedOptimizer(Optimizer.LAGDHC);
        } else if (optimizerString.equals(Optimizer.TABU.toString())) {
            setSelectedOptimizer(Optimizer.TABU);

        }
    }

    /**
     * Parse gui option, if given string is not parsable to Double default value is set.
     *
     * @param option       string to parse
     * @param defaultValue if given string is not parsable this value is set
     * @return
     */
    private double parseOptionDouble(String option, double defaultValue) {
        if (option.length() != 0) {
            return Double.parseDouble(option);
        } else {
            return defaultValue;
        }
    }

    /**
     * Parse gui option, if given string is not parsable to Integer default value is set.
     *
     * @param option       string to parse
     * @param defaultValue if given string is not parsable this value is set
     * @return
     */
    private int parseOptionInteger(String option, int defaultValue) {
        if (option.length() != 0) {
            return Integer.parseInt(option);
        } else {
            return defaultValue;
        }
    }

    /**
     * Parse gui option, if given string is not parsable to Long default value is set.
     *
     * @param option       string to parse
     * @param defaultValue if given string is not parsable this value is set
     * @return
     */
    private long parseOptionLong(String option, long defaultValue) {
        if (option.length() != 0) {
            return Long.parseLong(option);
        } else {
            return defaultValue;
        }
    }

    /**
     * Gets the current settings of the search algorithm.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        Vector<String> options = new Vector<String>();

        options.add("-B");
        options.add("" + getF_beta());

        options.add("-Q");
        options.add("" + getQ0());

        options.add("-X");
        options.add("" + getF_localUpdateCoef());

        options.add("-V");
        options.add("" + getF_globalUpdateCoef());

        options.add("-I");
        options.add("" + getNumOfIterations());

        options.add("-M");
        options.add("" + getNumOfAnts());

        options.add("-S");
        options.add("" + getSeed());

        options.add("-H");
        options.add("" + getOptimizationStep());

        options.add("-opt");
        options.add("" + getSelectedOptimizer());

        options.add("-P");
        options.add("" + getParallelize());

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    } // getOptions

    // setters and getters used in GUI

    public double getF_beta() {
        return f_beta;
    }

    public void setF_beta(double f_beta) {
        this.f_beta = f_beta;
    }

    public double getQ0() {
        return q0;
    }

    public void setQ0(double q0) {
        this.q0 = q0;
    }

    public double getF_localUpdateCoef() {
        return f_localUpdateCoef;
    }

    public void setF_localUpdateCoef(double f_localUpdateCoef) {
        this.f_localUpdateCoef = f_localUpdateCoef;
    }

    public double getF_globalUpdateCoef() {
        return f_globalUpdateCoef;
    }

    public void setF_globalUpdateCoef(double f_globalUpdateCoef) {
        this.f_globalUpdateCoef = f_globalUpdateCoef;
    }

    public int getNumOfIterations() {
        return numOfIterations;
    }

    public void setNumOfIterations(int numOfIterations) {
        this.numOfIterations = numOfIterations;
    }

    public int getNumOfAnts() {
        return numOfAnts;
    }

    public void setNumOfAnts(int numOfAnts) {
        this.numOfAnts = numOfAnts;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public long getSeed() {
        return seed;
    }

    public void setOptimizationStep(int optimizationStep) {
        this.optimizationStep = optimizationStep;
    }

    public int getOptimizationStep() {
        return optimizationStep;
    }


    public Optimizer getSelectedOptimizer() {
        return selectedOptimizer;
    }

    public void setSelectedOptimizer(Optimizer selectedOptimizer) {
        this.selectedOptimizer = selectedOptimizer;
    }

    public void setParallelize(boolean parallelize){
        this.parallelize = parallelize;
    }

    public boolean getParallelize(){
        return parallelize;
    }
    private enum Optimizer {
        HCST, LAGDHC, TABU
    }

}//class AntColonyOptimization
