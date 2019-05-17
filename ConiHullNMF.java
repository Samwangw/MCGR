package indi.wangwei.util.matrix.decomposation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

import indi.wangwei.util.matrix.Matrix;

public class ConiHullNMF {
	public static List<double[][]> decomposation(double[][] A, double subrat, int k) throws Exception{
		checkMatrix(A);
		double[][] anchors = getAnchors(A, subrat);
		if (k>anchors.length)
			return null;
		int row = A.length;
		int col = A[0].length;
		double[][] U,V;
		V = new double[k][col];
		for(int i=0;i<k;i++)
			for(int j=0;j<col;j++){
				V[i][j]= A[(int)anchors[i][1]][j];
			}
		U = new double[row][k];
		for(int i=0;i<row;i++)
			U[i]= Matrix.lsqnonneg(V,A[i]);
		List<double[][]> factors = new ArrayList<double[][]>();
		factors.add(U);
		factors.add(V);
		return factors;
	}

	public static double[][] getAnchors(double[][] A, double subrat) {
		int row = A.length;
		int col = A[0].length;
		double[][] index= new double[col][1];
		for(int i=0;i<col;i++)
			index[i][0]=i;
		double[] arrayTrilA = Matrix.matrix2array(Matrix.tril(Matrix.repmat(index, 1, col)),2);//transform TrilA to array column by column
		int[] arrayTrialA2 = removeNan(arrayTrilA);//remove 0 from arrayTrilA
		index = null;
		arrayTrilA = null;
		double[][] index2= new double[1][col];
		for(int i=0;i<col;i++)
			index2[0][i]=i;
		double[] arrayTrilB = Matrix.matrix2array(Matrix.tril(Matrix.repmat(index2, col, 1)),2);//transform TrilB to array column by column
		int[] arrayTrialB2 = removeNan(arrayTrilB);//remove 0 from arrayTrilB
		index2 = null;
		arrayTrilB = null;
		// statistics of the available pairwise angles in each 2-dim subspace
		double[][] id = toBinary(A);
		double[][] ids = Matrix.tril(Matrix.multiply(Matrix.transpose(id), id));
		double[][] w = new double[arrayTrialA2.length][2];
		for(int i=0;i<arrayTrialA2.length;i++){
			w[i][0] = ids[arrayTrialA2[i]][arrayTrialB2[i]];
			w[i][1] = i;
		}
		ids= null;
		Arrays.sort(w,new Comparator<double[]>(){
			@Override public int compare(final double[] o1, final double[] o2) {
				return Double.compare(o2[0], o1[0]);
			}
		});
		//		for(int i=0;i<arrayTrialA2.length;i++){
		//			System.out.println(i+":"+w[i][0]+","+w[i][1]);
		//		}
		//double[][] result = sort(w);

		//randmly choose samnum 2-dim subspaces for test
		int nzw = Matrix.nnz(Matrix.transpose(w)[0]);
		int samnum=(int) Math.round(subrat*nzw);
		double[] samidx = new double[samnum];
		double[] weight = new double[samnum];
		for(int i=0;i<samnum;i++){
			samidx[i] = w[i][1];
			weight[i] = w[i][0]/row;
		}
		w = null;
		double[][] mask;
		double[] nnzo = new double[row];
		mask = new double[row][samnum];
		for(int i=0;i<row;i++)
			for(int j=0;j<samnum;j++){
				mask[i][j]= id[i][arrayTrialA2[(int)samidx[j]]]*id[i][arrayTrialB2[(int)samidx[j]]];
				nnzo[i]+=mask[i][j];
			}
		id = null;

		double[][]xa = new double[row][samnum];
		double[][]xb = new double[row][samnum];
		for(int i=0;i<row;i++)
			for(int j=0;j<samnum;j++){
				xa[i][j]= A[i][arrayTrialA2[(int)samidx[j]]];
				xb[i][j]= A[i][arrayTrialB2[(int)samidx[j]]];
			}
		samidx = null;
		arrayTrialA2 = arrayTrialB2 = null;
		double[][]angles_m= new double[row][samnum]; 
		for(int i=0;i<row;i++)
			for(int j=0;j<samnum;j++){
				if(mask[i][j]>0)
					angles_m[i][j]= Math.atan2(xa[i][j], xb[i][j]);
				else
					angles_m[i][j]= Double.NaN;
			}
		mask = null;
		xa = xb = null;
		double[] maxangle,minangle;
		int[] maxid,minid;
		maxangle= new double[samnum];
		minangle= new double[samnum];
		maxid = new int[samnum];
		minid = new int[samnum];
		for(int i =0;i<samnum;i++){
			maxangle[i] = Integer.MIN_VALUE;
			minangle[i] = Integer.MAX_VALUE;
		}
		for(int i=0;i<row;i++)
			for(int j=0;j<samnum;j++){
				if(angles_m[i][j]>maxangle[j]){
					maxangle[j] = angles_m[i][j];
					maxid[j] = i;
				}
				if(angles_m[i][j]<minangle[j]){
					minangle[j] = angles_m[i][j];
					minid[j] = i;
				}
			}
		angles_m = null;
		double[][] sparse  = new double[row][2];
		for(int i=0;i<samnum;i++){
			int rowindex1 = maxid[i];
			sparse[rowindex1][0] += weight[i];
			int rowindex2 = minid[i];
			sparse[rowindex2][0] += weight[i];
		}
		weight = null;
		for(int i=0;i<row;i++){
			sparse[i][0] /= nnzo[i];
			sparse[i][1]=i;
		}
		maxangle= minangle=null;
		maxid = minid = null;
		Arrays.sort(sparse,new Comparator<double[]>(){
			@Override public int compare(final double[] o1, final double[] o2) {
				return Double.compare(o2[0], o1[0]);
			}
		});
		return sparse;
	}

	public static int[] removeNan(double[] old){
		if (old == null)
			return null;
		int index =0;
		double[] a = old.clone();
		for(int i=0;i<a.length;i++)
		{
			if(!Double.isNaN(a[i])){
				a[index] = a[i];
				index++;
			}
		}
		int[]b = new int[index];
		for(int i=0;i<index;i++)
			b[i] = (int)a[i];
		return b;
	}
	public static double[][] toBinary(double[][] A){
		double[][] id = new double[A.length][A[0].length];
		for(int i=0;i<id.length;i++)
			for(int j=0;j<id[0].length;j++)
				if(A[i][j]!=0)
					id[i][j]=1;
		return id;
	}
	public static void checkMatrix(double[][]a) throws Exception{
		if(a == null)
			throw new Exception("Null matrix is inputed");
		for(int i=0;i<a.length;i++)
			for(int j=0;j<a[0].length;j++)
				if(a[i][j]<0)
					throw new Exception("Matrix element [" + i+","+j+"] is negative.");
	}

}
