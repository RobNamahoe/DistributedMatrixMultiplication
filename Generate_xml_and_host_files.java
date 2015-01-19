import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.PriorityQueue;

public class Generate_xml_and_host_files {

  File dir;
  String xmlfilepath, hostfilepath;

  static int latency = 20;
  static int bandwidth = 10;
  
  static int N, num_hosts;
  static HashMap<Integer, Node> nodes;
  static HashMap<String, String> links;

  public static void main(String[] args) {
    
    if (args.length == 0) {
      System.err.println("Error: Invalid arguments. Usage: java Generate_xml_and_host_files <num_hosts> [optional: latency] [optional: bandwidth]");
      System.exit(-1);
    }
    else if (args.length == 2) {
      latency = Integer.parseInt(args[1]);
    }
    else if (args.length == 3) {
      latency = Integer.parseInt(args[1]);
      bandwidth = Integer.parseInt(args[2]);
    }
    
    Generate_xml_and_host_files writer = new Generate_xml_and_host_files(Integer.parseInt(args[0]));

    writer.createHostFile();
    writer.createXmlFile();

  }

  /**
   * Default constructor method
   * @param num_hosts Number of hosts
   */
  public Generate_xml_and_host_files(int num_hosts) {

    this.num_hosts = num_hosts;
    this.N = (int) Math.sqrt(num_hosts);
    
    if (N * N != num_hosts) {
      System.err.println("Error: Number of hosts must be a perfect square.");
      System.exit(-1);
    }
    
    nodes = new HashMap<Integer, Node>();
    links = new HashMap<String, String>();

    dir = new File(".");
    try {
      hostfilepath = dir.getCanonicalPath() + File.separator + "hostfile_" + num_hosts;
      xmlfilepath = dir.getCanonicalPath() + File.separator + "2Dtorus_" + num_hosts + ".xml";
    }
    catch (IOException e) { /* Do nothing */
    }

    // Create Nodes and determine link names
    for (int i = 0; i < num_hosts; i++) {
      nodes.put(i, new Node(i));
      links.put(i + "-" + getRight(i), null);
      links.put(i + "-" + getDown(i), null);
    }

    // Set neighbors
    Node node;
    for (int rank : nodes.keySet()) {
      node = nodes.get(rank);
      node.setUp(nodes.get(getUp(rank)));
      node.setDown(nodes.get(getDown(rank)));
      node.setLeft(nodes.get(getLeft(rank)));
      node.setRight(nodes.get(getRight(rank)));
    }

  }

  public void createHostFile() {
    FileWriter fstream = null;
    BufferedWriter out = null;

    try {
      fstream = new FileWriter(hostfilepath);
      out = new BufferedWriter(fstream);

      for (int i = 0; i < num_hosts; i++) {
        out.write("host-" + i + ".hawaii.edu");
        out.newLine();
      }
      out.close();
    }
    catch (IOException e) {
      /* Do nothing */
    }

  }

  public void createXmlFile() {

    printHeader();

    // Print routes
    for (int i = 0; i < num_hosts; i++) {
      SingleSourceShortestPath(i);
      for (int j = i + 1; j < num_hosts; j++) {
        printPath(j, i);
      }
    }

    printFooter();

  }

  public void printPath(int src, int dst) {
    FileWriter fstream = null;
    BufferedWriter out = null;

    try {
      fstream = new FileWriter(xmlfilepath, true);
      out = new BufferedWriter(fstream);

      out.write("  <route src=\"host-" + src + ".hawaii.edu\" dst=\"host-" + dst + ".hawaii.edu\">");
      out.newLine();

      String temp;
      Node pi = null;
      Node node = nodes.get(src);
      while (node.getPi() != null) {
        pi = node.getPi();
        temp = node.getRank() + "-" + pi.getRank();
        if (links.containsKey(temp)) {
          out.write(" <link_ctn id=\"link-" + node.getRank() + "-" + pi.getRank() + "\"/>");
        }
        else if (links.containsKey(pi.getRank() + "-" + node.getRank())) {
          out.write(" <link_ctn id=\"link-" + pi.getRank() + "-" + node.getRank() + "\"/>");
        }
        else {
          System.err.println("Link does not exist");
        }
        out.newLine();
        node = pi;
      }

      out.write("  </route>");
      out.newLine();

      out.close();

    }
    catch (IOException e) {
      /* Do nothing */
    }
  }

  public void printHeader() {

    FileWriter fstream = null;
    BufferedWriter out = null;

    try {
      fstream = new FileWriter(xmlfilepath);
      out = new BufferedWriter(fstream);

      out.write("<?xml version='1.0'?>");
      out.newLine();
      out.write("<!DOCTYPE platform SYSTEM \"http://simgrid.gforge.inria.fr/simgrid.dtd\">");
      out.newLine();
      out.write("<platform version=\"3\">");
      out.newLine();
      out.write("<AS id=\"AS0\" routing=\"Full\">");
      out.newLine();
      out.newLine();

      // Print hosts
      for (int i = 0; i < num_hosts; i++) {
        out.write("  <host id=\"host-" + i + ".hawaii.edu\" power=\"1\"/>");
        out.newLine();
      }

      out.newLine();

      // Print Links
      for (int i = 0; i < num_hosts; i++) {
        out.write("  <link id=\"link-" + i + "-" + getRight(i) + "\" latency=\"" + latency + "us\" bandwidth=\"" + bandwidth + "Gbps\"/>");
        out.newLine();
        out.write("  <link id=\"link-" + i + "-" + getDown(i) + "\" latency=\"" + latency + "us\" bandwidth=\"" + bandwidth + "Gbps\"/>");
        out.newLine();
      }

      out.newLine();
      out.close();
    }
    catch (IOException e) {
      /* Do nothing */
    }
  }

  public void printFooter() {
    FileWriter fstream = null;
    BufferedWriter out = null;

    try {
      fstream = new FileWriter(xmlfilepath, true);
      out = new BufferedWriter(fstream);

      out.write("</AS>");
      out.newLine();
      out.write("</platform>");
      out.newLine();

      out.close();

    }
    catch (IOException e) {
      /* Do nothing */
    }
  }

  public void SingleSourceShortestPath(int rank) {
    Dijkstra(rank);
  }

  private void Dijkstra(int rank) {

    ArrayList<Node> S = new ArrayList<Node>();
    PriorityQueue<Node> Q = new PriorityQueue<Node>(num_hosts, new NodeComparator());

    initializeSingleSource(rank);

    Q.add(nodes.get(rank));

    while (!Q.isEmpty()) {
      Node u = Q.poll();
      S.add(u);
      Iterator<Node> it = u.getNodes();

      while (it.hasNext()) {
        Node v = it.next();
        relax(u, v);
        if (!S.contains(v)) {
          Q.remove(v);
          Q.add(v);
        }
      }
    }
  }

  private class NodeComparator implements Comparator<Node> {

    @Override
    public int compare(Node node1, Node node2) {
      return (int) (node2.getD() - node1.getD());
    }
  }

  private static void initializeSingleSource(int source_rank) {
    for (Node node : nodes.values()) {
      node.setD(Double.POSITIVE_INFINITY);
      node.setPi(null);
    }
    Node node = nodes.get(source_rank);
    node.setD(0.0);
    nodes.put(source_rank, node);
  }

  private void relax(Node u, Node v) {
    double vd = v.getD();
    double ud = u.getD();
    if (vd > ud + 1) {
      v.setD(ud + 1);
      v.setPi(u);
    }
  }

  public static int getRight(int index) {
    return index + 1 - ((index % N == (N - 1)) ? N : 0);
  }

  public static int getDown(int index) {
    int neighbor = index + N;
    if (neighbor >= num_hosts) {
      neighbor = neighbor - num_hosts;
    }
    return neighbor;
  }

  public static int getUp(int index) {
    int neighbor = index - N;
    if (neighbor < 0) {
      neighbor = neighbor + num_hosts;
    }
    return neighbor;
  }

  public static int getLeft(int index) {
    return index - 1 + ((index % N == 0) ? N : 0);
  }

  /**
   * Private nested Node class
   */
  private class Node {

    private int rank;

    // Used in Dijkstra's Algorithm to find shortest paths
    private Node pi;
    private double d;

    // Neighbors
    private Node up, down, left, right;

    // Constructor
    public Node(int rank) {
      this.rank = rank;
    }

    public void setUp(Node node) {
      this.up = node;
    }

    public void setDown(Node node) {
      this.down = node;
    }

    public void setLeft(Node node) {
      this.left = node;
    }

    public void setRight(Node node) {
      this.right = node;
    }

    public void setPi(Node node) {
      this.pi = node;
    }

    public void setD(double d) {
      this.d = d;
    }

    public int getRank() {
      return this.rank;
    }

    public Node getUp() {
      return this.up;
    }

    public Node getDown() {
      return this.down;
    }

    public Node getLeft() {
      return this.left;
    }

    public Node getRight() {
      return this.right;
    }

    public Node getPi() {
      return this.pi;
    }

    public double getD() {
      return this.d;
    }

    public Iterator<Node> getNodes() {
      ArrayList<Node> nodes = new ArrayList<Node>();
      nodes.add(this.up);
      nodes.add(this.left);
      nodes.add(this.down);
      nodes.add(this.right);
      return nodes.iterator();
    }

  }

}
