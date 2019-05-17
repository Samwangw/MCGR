package net.ww.rs.groupAlgorithms;

import indi.wangwei.util.matrix.decomposation.ConiHullNMF;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import net.ww.rs.Group;
import net.ww.rs.Item;
import net.ww.rs.Items;
import net.ww.rs.Rating;
import net.ww.rs.User;
import net.ww.rs.Users;
import net.ww.rs.similarity.EntropyImportance;
import net.ww.rs.similarity.PCC;

public class SNMF {
	public  String name;
	public Group group = new Group();
	public  Users users = new Users();
	public  Items items = new Items();
	public  LinkedList<Rating> tasks = new LinkedList<Rating>();
	public  LinkedList<Rating> finishedtasks = new LinkedList<Rating>();
	public  LinkedList<Rating> predictions = new LinkedList<Rating>();

	public void load(Group group, Users users, Items items, List<Rating> tasks){
		this.name = "SNMF";
		for(User u:group)
			this.group.addMember(u);
		for(User u:users)
			this.users.addUser(u);
		for(Item i:items)
			this.items.addItem(i);
		for(Rating r:tasks)
			this.tasks.add(r);
	}
	private User cast2User(Group group) throws Exception{
		if (group == null)
			return null;
		double[][] ratingMatrix = group.getRatingMatrix();
		double[][] anchors = ConiHullNMF.getAnchors(ratingMatrix, 0.10);
		double[] weight = new double[group.size()];
		for(int i=0;i<anchors.length;i++){
			int index = (int)anchors[i][1];
			double w = anchors[i][0];
			if(Double.isNaN(w))
				w=0;
			weight[index]= w;
		}
		double sum_weight =0;
		for(int i=0;i<weight.length;i++){
			sum_weight += weight[i];
		}
		for(int i=0;i<weight.length;i++){
			weight[i]/= sum_weight;
		}
		//store weights of members
		ConcurrentHashMap<Integer, Double> weights = new ConcurrentHashMap<Integer, Double>();
		Iterator<User> iter2 = group.iterator();
		while(iter2.hasNext()){
			int index =0;
			User u = iter2.next();
			weights.put(u.getId(),weight[index++]);
		}
		
		ConcurrentHashMap<Integer, Double> tmp_weights = new ConcurrentHashMap<Integer, Double>();
		ConcurrentHashMap<Integer, Double> tmp_values = new ConcurrentHashMap<Integer, Double>();
		//build group user
		User user =  new User(-1);
		Iterator<User> iter = group.iterator();
		int index = -1;
		while(iter.hasNext()){
			User u = iter.next();
			index++;
			for(Rating r:u.getRatings().values()){
				Item item = r.getItem();
				double value = r.getValue();
				if(tmp_weights.containsKey(item.getId())){
					double p_weight = tmp_weights.get(item.getId());
					double p_value = tmp_values.get(item.getId());
					tmp_weights.put(item.getId(), p_weight+weights.get(u.getId()));
					tmp_values.put(item.getId(), p_value+value*weights.get(u.getId()));
				}
				else{
					tmp_weights.put(item.getId(),  weights.get(u.getId()));
					tmp_values.put(item.getId(),  weights.get(u.getId())*value);
				}
			}
		}
		System.out.println("pseudo user:"+tmp_weights.keySet().size()+" profile");
		if (tmp_weights.keySet().size()>50)
			return null;
		for(Integer id:tmp_weights.keySet()){
			
			user.addRating(new Rating(user,items.getItembyID(id),tmp_values.get(id)/tmp_weights.get(id)));
			System.out.println("item id:"+id +"  value:"+tmp_values.get(id)/tmp_weights.get(id));
		}
		return user;
	}
	public List<Rating> cmpPredictions() throws Exception{
		Iterator<Rating> iter = tasks.iterator();
		User active_user = cast2User(group);
		if(active_user != null){
			int index =0;
			while(iter.hasNext()){
				Rating task = iter.next();
				if(!group.contain(task.getUser()))
					continue;
				Item target_item = task.getItem();
				double pre = cmpPrediction(active_user, target_item);
				Rating r = new Rating(active_user,target_item,pre);
				finishedtasks.add(task);
				predictions.add(r);
				index++;
				//System.out.println(index+" task finished.");
			}
		}
		return predictions;
	}
	public double cmpPrediction(User activeUser, Item item){
		double pre = Double.NaN;
		if (item == null)
			return pre;
		if (activeUser == null)
			return pre;
		if(activeUser.getRatingByItem(item.getId())!=null)
			return pre;
		if (item.getRatings().size()>0)
			pre =0;
		else
			return pre;
		double sumSim =0;
		for (Rating r: item.getRatings().values()){
			User u = r.getUser();
			double sim = cmpSim(activeUser,u);
			if (!Double.isNaN(sim)){
				sumSim+= Math.abs(sim);
				pre += sim *(r.getValue()-u.getMeanRating());
			}
		}
		double avg =0;
		//avg = activeUser.getMeanRating();
		avg = activeUser.getApproMeanRating(item, 0.4);
		pre = avg+pre/sumSim;
		if(pre>Rating.MAX_RATING)
			pre = Rating.MAX_RATING;
		else if(pre<Rating.MIN_RATING)
			pre = Rating.MIN_RATING;
		return pre;
	}
	public double cmpSim(User user1, User user2){
		double sim = Double.NaN;
		double[][] co = user1.getCoRatings(user2);
		if(co!=null)
			sim =0;
		else 
			return sim;
		//sim = EntropyImportance.cmpSim(co[0], co[1], 0.7);
		sim = PCC.cmpSim(co[0], co[1], 1);
		return sim;
	}
}
