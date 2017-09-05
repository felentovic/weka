package weka.classifiers.bayes.net.search.local;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.net.ParentSet;
import weka.core.*;

import java.util.*;

/**
 * Created by felentovic on 26/08/17.
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
         * product of pheromone value and heuristic value for arc
         **/
        double[][] m_arcPheromoneHeuristic;

        /**
         * number of arcs which are still candidates ( not in G and inclusion of arc improves G)
         */
        int numOfAvailableArcs;
        /**
         * sum of cells that are still candidates  ( not in G and inclusion of arc improves G)
         **/
        double sumOfPheromoneHeuristic;

        /**
         * c'tor
         *
         * @param nNrOfNodes number of nodes in network, used to determine memory
         *                   size to reserve
         */
        Cache(int nNrOfNodes) {
            m_fDeltaScoreAdd = new double[nNrOfNodes][nNrOfNodes];
            m_arcPheromoneHeuristic = new double[nNrOfNodes][nNrOfNodes];
        }

        /**
         * set cache entry
         *
         * @param m_nHead number of head node
         * @param m_nTail number of tail node
         * @param fValue  value to put in cache
         */
        public void putScore(int m_nTail, int m_nHead, double fValue) {
            m_fDeltaScoreAdd[m_nTail][m_nHead] = fValue;
        } // put

        /**
         * get cache entry
         *
         * @return cache value
         */
        public double getScore(int m_nTail, int m_nHead) {
            return m_fDeltaScoreAdd[m_nTail][m_nHead];

        } // get

        /**
         * set cache entry
         *
         * @param m_nHead number of head node
         * @param m_nTail number of tail node
         * @param fValue  value to put in cache
         */
        public void putPheromoneHeuristic(int m_nTail, int m_nHead, double fValue) {
            m_arcPheromoneHeuristic[m_nTail][m_nHead] = fValue;
        } // put

        /**
         * get cache entry
         *
         * @return cache value
         */
        public double getPheromoneHeuristic(int m_nTail, int m_nHead) {
            return m_arcPheromoneHeuristic[m_nTail][m_nHead];
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

    private class Ant extends LocalScoreSearchAlgorithm{
        /**
         * cache for storing score differences
         **/
        Cache m_Cache = null;
        /**
         * exponent of pheromone cell in formula (11)
         **/
        double f_alfa;
        /**
         * exponent of diff score cell in formula (11)
         **/
        double f_beta;
        /**
         * selection probability  formula (10)
         **/
        double q0;
        /**
         * initial pheromone level
         **/
        double pheromone0;
        /**
         * coefficient in local pheromone update
         **/
        double explorationCoeff;
        /**
         * true if there is arc (tail,head)
         */
        boolean[][] m_arcs;

        @Override
        public void search(BayesNet bayesNet, Instances instances) throws Exception {
            m_arcs = new boolean[instances.numAttributes()][instances.numAttributes()];
            initCache(bayesNet, instances);

            do {
                int[] indices = selectIndices(bayesNet, instances, q0);
                int attributeTail = indices[0];
                int attributeHead = indices[1];

                if (attributeTail == -1 || attributeHead == -1) {
                    break;
                }
                //add Tail in parent set of Head
                ParentSet parentSet = bayesNet.getParentSet(attributeHead);
                parentSet.addParent(attributeTail, instances);
                // set value to used
                m_Cache.putScore(attributeTail, attributeHead, Double.NEGATIVE_INFINITY);
                m_arcs[attributeTail][attributeHead] = true;

                updateAncestorDescendantArcs(attributeTail, attributeHead, bayesNet, instances);
                updateCacheMatrices(attributeHead, instances.numAttributes(), parentSet);
                updatePheromoneHeuristicCache(bayesNet, instances);

                //local pheromone update
                pheromone[attributeTail][attributeHead] = (1 - explorationCoeff) * pheromone[attributeTail][attributeHead]
                        + explorationCoeff * pheromone0;
            } while (m_Cache.numOfAvailableArcs > 0);
        }

        /**
         * Select two indices (arc) from pair candidates. With probability q0 arc that gives the most improvement is selected,
         * and with probability (1 - q0) arc is selected regarding its probability. Probability is calculated using formula (11)
         * in the paper.
         *
         * @param bayesNet  Bayes network to be learned
         * @param instances data set to learn from
         * @param q0        probability of method for selecting arcs
         * @return
         */
        int[] selectIndices(BayesNet bayesNet, Instances instances, double q0) {
            int[] arcs;
            if (Math.random() < q0) {
                arcs = findBestArc(bayesNet, instances);
            } else {
                arcs = proportionallySelectIndices(bayesNet, instances);
            }
            return arcs;
        }

        /**
         * find best (or least bad) arc regarding product of score diff and pheromone value powered by f_beta coef
         *
         * @param bayesNet  Bayes network to add arc to
         * @param instances data set
         * @return Array of ints where on first index is tail and on second head, or {-1,-1} if no arc addition is
         * allowed (this can happen if any arc addition introduces a cycle, or
         * all parent sets are filled up to the maximum nr of parents).
         */
        private int[] findBestArc(BayesNet bayesNet, Instances instances) {
            int[] bestArc = new int[]{-1, -1};
            double bestScore = -1;
            int nNrOfAtts = instances.numAttributes();
            // find best arc
            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                if (bayesNet.getParentSet(iAttributeHead).getNrOfParents() < m_nMaxNrOfParents) {
                    for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                        if (addArcMakesSense(bayesNet, instances, iAttributeHead,
                                iAttributeTail)) {
                            double tmp = Math.pow(m_Cache.getScore(iAttributeTail, iAttributeHead), f_beta) * pheromone[iAttributeTail][iAttributeHead];
                            if (tmp > bestScore) {
                                bestArc[0] = iAttributeTail;
                                bestArc[1] = iAttributeHead;
                                bestScore = tmp;
                            }
                        }
                    }
                }
            }
            return bestArc;
        }//findBestArc

        /**
         * Selects arc proportionally using probability calculated using formula (11) in paper
         *
         * @param bayesNet  Bayes network to add arc to
         * @param instances data set
         * @return
         */
        private int[] proportionallySelectIndices(BayesNet bayesNet, Instances instances) {
            int[] indices = new int[]{-1, -1};
            double randValue = Math.random() * m_Cache.sumOfPheromoneHeuristic;

            int nNrOfAtts = instances.numAttributes();
            double counter = 0;
            for (int iAttributeHead = 0, index = 0; iAttributeHead < nNrOfAtts; iAttributeHead++, index++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    double tmpPhHeu = m_Cache.getPheromoneHeuristic(iAttributeTail, iAttributeHead);
                    if (Double.compare(tmpPhHeu, 0) > 0 && addArcMakesSense(bayesNet, instances, iAttributeHead, iAttributeTail)) {
                        counter += tmpPhHeu;
                        if (Double.compare(randValue, counter) <= 0) {
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
         * update the cache due to change of parent set of a node
         *
         * @param iAttributeHead node that has its parent set changed
         * @param nNrOfAtts      number of nodes/attributes in data set
         * @param parentSet      new parents set of node iAttributeHead
         */
        private void updateCacheMatrices(int iAttributeHead, int nNrOfAtts, ParentSet parentSet) {
            // update score cache entries for arrows heading towards iAttributeHead
            double fBaseScore = calcNodeScore(iAttributeHead);
            int nNrOfParents = parentSet.getNrOfParents();
            for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                //if there is not an arc already
                if (Double.compare(m_Cache.getScore(iAttributeTail, iAttributeHead), Double.NEGATIVE_INFINITY) != 0) {
                    // add entries to cache for adding arcs
                    if (nNrOfParents < m_nMaxNrOfParents) {
                        double valScore = calcScoreWithExtraParent(iAttributeHead, iAttributeTail)
                                - fBaseScore;
                        m_Cache.putScore(iAttributeTail, iAttributeHead, valScore);
                        double valPhHeu = Math.pow(pheromone[iAttributeTail][iAttributeHead], f_alfa) * Math.pow(valScore, f_beta);
                        m_Cache.putPheromoneHeuristic(iAttributeTail, iAttributeHead, valPhHeu);

                    }
                }
            }
        } // updateCacheMatrices

        /**
         * Updates number of available arcs and sum of cells of candidate arcs.
         *
         * @param bayesNet  Bayes network to add arc to
         * @param instances data set
         */
        private void updatePheromoneHeuristicCache(BayesNet bayesNet, Instances instances) {
            int nNrOfAtts = instances.numAttributes();
            double tmpSum = 0;
            int tmpAvailableArcs = 0;
            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    //if it creates a cycle or valScore is < 0 then it is not in set Fg
                    if (m_Cache.getScore(iAttributeTail, iAttributeHead) < 0 || !addArcMakesSense(bayesNet, instances, iAttributeHead, iAttributeTail)) {
                        m_Cache.putPheromoneHeuristic(iAttributeTail, iAttributeHead, -1);

                    } else {
                        //sum all positive values
                        tmpSum += m_Cache.getPheromoneHeuristic(iAttributeTail, iAttributeHead);
                        tmpAvailableArcs += 1;
                    }
                }
            }
            m_Cache.sumOfPheromoneHeuristic = tmpSum;
            m_Cache.numOfAvailableArcs = tmpAvailableArcs;
        }//updatePheromoneHeuristicCache

        /**
         * initCache initializes the cache
         *
         * @param bayesNet  Bayes network to be learned
         * @param instances data set to learn from
         * @throws Exception if something goes wrong
         */

        private void initCache(BayesNet bayesNet, Instances instances) throws Exception {
            // determine base scores
            double[] fBaseScores = new double[instances.numAttributes()];
            int nNrOfAtts = instances.numAttributes();

            m_Cache = new Cache(nNrOfAtts);
            for (int iAttribute = 0; iAttribute < nNrOfAtts; iAttribute++) {
                fBaseScores[iAttribute] = calcNodeScore(iAttribute);
            }

            for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                for (int iAttributeTail = 0; iAttributeTail < nNrOfAtts; iAttributeTail++) {
                    double valScore;
                    double valPhHeu;
                    if (iAttributeHead == iAttributeTail) {
                        valScore = Double.NEGATIVE_INFINITY;
                        valPhHeu = 0;
                    } else {
                        valScore = calcScoreWithExtraParent(iAttributeHead, iAttributeTail)
                                - fBaseScores[iAttributeHead];
                        valPhHeu = Math.pow(pheromone[iAttributeTail][iAttributeHead], f_alfa) * Math.pow(valScore, f_beta);

                    }
                    m_Cache.putScore(iAttributeTail, iAttributeHead, valScore);
                    m_Cache.putPheromoneHeuristic(iAttributeTail, iAttributeHead, valPhHeu);
                }
            }
            updatePheromoneHeuristicCache(bayesNet, instances);
            // m_Cache.numOfAvailableArcs = nNrOfAtts * (nNrOfAtts - 1) should be;

        }// initCache


        /**
         * Move all ancestors of Tail and descendants of Head from candidates list
         *
         * @param attributeTail tail of arc
         * @param attributeHead head of arc
         * @param bayesNet      Bayes network to be learned
         * @param instances     data set to learn from
         */
        private void updateAncestorDescendantArcs(int attributeTail, int attributeHead, final BayesNet bayesNet, Instances instances) {
            //all ancestors of AttributeTail
            List<Integer> ancestors = imitateRecursion(attributeTail, instances.numAttributes() / 2, new Function() {
                @Override
                public void execute(ArrayList<Integer> list, int iNode) {
                    int[] parents = bayesNet.getParentSet(iNode).getParents();
                    for (int i = 0; i < parents.length; i++) {
                        list.add(parents[i]);
                    }
                }
            });
            //all descendants of AttributeHead
            List<Integer> descendants = imitateRecursion(attributeHead, instances.numAttributes() / 2, new Function() {
                @Override
                public void execute(ArrayList<Integer> list, int iNode) {
                    boolean[] descedants = m_arcs[iNode];
                    for (int iHead = 0; iHead < descedants.length; iHead++) {
                        if (descedants[iHead]) {
                            list.add(iHead);
                        }
                    }
                }
            });

            for (Integer iTail : ancestors) {
                for (Integer iHead : descendants) {
                    m_Cache.putScore(iTail, iHead, Double.NEGATIVE_INFINITY);
                }
            }

        }//updateAncestorDescendantArcs


        /**
         * Imitates recursive visit of tree
         *
         * @param initialValue start node
         * @param initialSize  initial size of list
         * @param function     the way in which next level of nodes is generated
         * @return list of all visited nodes
         */
        private List<Integer> imitateRecursion(int initialValue, int initialSize, Function function) {
            ArrayList<Integer> list = new ArrayList<>(initialSize);
            list.add(initialValue);
            ArrayList<Integer> listPrev = new ArrayList<>(initialSize);
            ArrayList<Integer> listCurr = new ArrayList<>(initialSize);
            listPrev.add(initialValue);

            while (!listPrev.isEmpty()) {
                for (Integer iNode : listPrev) {
                    function.execute(listPrev, iNode);
                }
                list.addAll(listCurr);
                listPrev.clear();
                listPrev.addAll(listCurr);
                listCurr.clear();
            }
            return list;
        }//imitateRecursion

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

        public void setF_alfa(double f_alfa) {
            this.f_alfa = f_alfa;
        }

        public void setF_beta(double f_beta) {
            this.f_beta = f_beta;
        }

        public void setQ0(double q0) {
            this.q0 = q0;
        }

        public void setPheromone0(double pheromone0) {
            this.pheromone0 = pheromone0;
        }

        public void setExplorationCoeff(double explorationCoeff) {
            this.explorationCoeff = explorationCoeff;
        }
    }//class Ant

    /**
     * Interface used in imitate recursion method
     */
    private interface Function {
        void execute(ArrayList<Integer> list, int iNode);
    }


    /**
     * pheromone level matrix
     */
    private double[][] pheromone;
    private double evaporationCoeff;
    private HillClimber hillClimber;

    private int numOfIterations;
    private int numOfAnts;

    /**
     * exponent of pheromone cell in formula (11)
     **/
    private double f_alfa;
    /**
     * exponent of diff score cell in formula (11)
     **/
    private double f_beta;
    /**
     * selection probability  formula (10)
     **/
    private double q0;
    /**
     * initial pheromone level
     **/
    private double pheromone0;
    /**
     * coefficient in local pheromone update
     **/
    private double explorationCoeff;

    @Override
    protected void search(BayesNet bayesNet, Instances instances) throws Exception {
        hillClimber = new HillClimber();
        hillClimber.setInitAsNaiveBayes(false);
        hillClimber.setMaxNrOfParents(m_nMaxNrOfParents);
        hillClimber.setUseArcReversal(true);
        hillClimber.buildStructure(bayesNet,instances);

        double totalScore = 0;
        for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++){
            totalScore += hillClimber.calcNodeScore(iAttribute);
        }

        //init pheromone matrix
        pheromone0 = 1 / Math.abs(totalScore);
        pheromone = new double[instances.numAttributes()][instances.numAttributes()];
        for (int i = 0; i < instances.numAttributes(); i++) {
            for (int j = 0; j < instances.numAttributes(); j++) {
                pheromone[i][j] = pheromone0;
            }
        }

        // keeps track of best structure found so far
        BayesNet bestBayesNet;

        // initialize bestBayesNet
        double fBestScore = totalScore;
        bestBayesNet = new BayesNet();
        bestBayesNet.m_Instances = instances;
        bestBayesNet.initStructure();
        copyParentSets(bestBayesNet, bayesNet);

        Ant ant = new Ant();
        initiliazeAnt(ant);

        BayesNet currentBayesNet = new BayesNet();
        currentBayesNet.m_Instances = instances;
        int iterationStep = 10;
        for (int iteration = 0; iteration < numOfIterations; iteration++) {
            for (int antNum = 0; antNum < numOfAnts; antNum++) {
                //new bayes net
                currentBayesNet.initStructure();
                ant.buildStructure(currentBayesNet, instances);
                //do local search on iteration step
                if (iteration % iterationStep == 0) {
                    hillClimber.buildStructure(currentBayesNet, instances);
                }
                double fCurrentScore = 0;
                for (int iAttribute = 0; iAttribute < instances.numAttributes(); iAttribute++) {
                    fCurrentScore += ant.calcNodeScore(iAttribute);
                }
                if (fCurrentScore >= fBestScore) {
                    fBestScore = fCurrentScore;
                    copyParentSets(bestBayesNet, currentBayesNet);
                }
            }

            //global pheromone update
            double reciprocalScore = 1 / fBestScore;
            for (int iAttributeTail = 0, nNrOfAtts = instances.numAttributes(); iAttributeTail < nNrOfAtts; iAttributeTail++) {
                for (int iAttributeHead = 0; iAttributeHead < nNrOfAtts; iAttributeHead++) {
                    if (ant.m_arcs[iAttributeTail][iAttributeHead]) {
                        pheromone[iAttributeTail][iAttributeHead] = (1 - evaporationCoeff) * pheromone[iAttributeTail][iAttributeHead]
                                + evaporationCoeff * reciprocalScore;
                    }
                }
            }
        }
        // restore current network to best network
        copyParentSets(bayesNet,bestBayesNet);
    }

    private void initiliazeAnt(Ant ant){
        //pheromone matrix and stuff
        ant.setMaxNrOfParents(m_nMaxNrOfParents);
        ant.setInitAsNaiveBayes(false);
        ant.setExplorationCoeff(explorationCoeff);
        ant.setF_alfa(f_alfa);
        ant.setF_beta(f_beta);
        ant.setPheromone0(pheromone0);
        ant.setQ0(q0);

    }

    /**
     * copyParentSets copies parent sets of source to dest BayesNet
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

        newVector.addElement(new Option("\tMaximum number of parents", "P", 1,
                "-P <nr of parents>"));
        newVector.addElement(new Option(
                "\tUse arc reversal operation.\n\t(default false)", "R", 0, "-R"));
        newVector.addElement(new Option(
                "\tInitial structure is empty (instead of Naive Bayes)", "N", 0, "-N"));
        newVector.addElement(new Option(
                "\tInitial structure specified in XML BIF file", "X", 1, "-X"));

        newVector.addAll(Collections.list(super.listOptions()));

        return newVector.elements();
    } // listOptions

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre>
     * -P &lt;nr of parents&gt;
     *  Maximum number of parents
     * </pre>
     *
     * <pre>
     * -R
     *  Use arc reversal operation.
     *  (default false)
     * </pre>
     *
     * <pre>
     * -N
     *  Initial structure is empty (instead of Naive Bayes)
     * </pre>
     *
     * <pre>
     * -mbc
     *  Applies a Markov Blanket correction to the network structure,
     *  after a network structure is learned. This ensures that all
     *  nodes in the network are part of the Markov blanket of the
     *  classifier node.
     * </pre>
     *
     * <pre>
     * -S [BAYES|MDL|ENTROPY|AIC|CROSS_CLASSIC|CROSS_BAYES]
     *  Score type (BAYES, BDeu, MDL, ENTROPY and AIC)
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        setUseArcReversal(Utils.getFlag('R', options));

        setInitAsNaiveBayes(!(Utils.getFlag('N', options)));

        m_sInitalBIFFile = Utils.getOption('X', options);

        String sMaxNrOfParents = Utils.getOption('P', options);
        if (sMaxNrOfParents.length() != 0) {
            setMaxNrOfParents(Integer.parseInt(sMaxNrOfParents));
        } else {
            setMaxNrOfParents(100000);
        }

        super.setOptions(options);
    } // setOptions

    /**
     * Gets the current settings of the search algorithm.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    @Override
    public String[] getOptions() {

        Vector<String> options = new Vector<String>();

        if (getUseArcReversal()) {
            options.add("-R");
        }

        if (!getInitAsNaiveBayes()) {
            options.add("-N");
        }
        if (m_sInitalBIFFile != null && !m_sInitalBIFFile.equals("")) {
            options.add("-X");
            options.add(m_sInitalBIFFile);
        }

        options.add("-P");
        options.add("" + m_nMaxNrOfParents);

        Collections.addAll(options, super.getOptions());

        return options.toArray(new String[0]);
    } // getOptions


    public double getF_alfa() {
        return f_alfa;
    }

    public void setF_alfa(double f_alfa) {
        this.f_alfa = f_alfa;
    }

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

    public double getExplorationCoeff() {
        return explorationCoeff;
    }

    public void setExplorationCoeff(double explorationCoeff) {
        this.explorationCoeff = explorationCoeff;
    }

    public double getEvaporationCoeff() {
        return evaporationCoeff;
    }

    public void setEvaporationCoeff(double evaporationCoeff) {
        this.evaporationCoeff = evaporationCoeff;
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
}//class AntColonyOptimization
