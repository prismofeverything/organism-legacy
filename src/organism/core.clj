(ns organism.core)

(defn map-cat
  [f s]
  (reduce into [] (mapv f s)))

(defn build-ring
  [symmetry color level]
  (mapv
   (fn [step]
     [color step])
   (range (* level symmetry))))

(defn build-rings
  [symmetry colors]
  (let [core-color (first colors)
        core (list [core-color 0])]
    (concat
     [[core-color core]]
     (mapv
      (fn [color level]
        [color
         (build-ring
          symmetry
          color
          level)])
      (rest colors)
      (map inc (range))))))

(defn rings->spaces
  [rings]
  (apply
   concat
   (mapv second rings)))

(defn space-adjacencies
  [rings level num-spaces space]
  (let [[color step] space
        neighbors (mapv
                   (fn [relation]
                     [color (mod (+ step relation) num-spaces)])
                   [-1 1])
        along (mod step level)
        axis? (zero? along)
        cycle (quot step level)
        inner-ring (nth rings (dec level))
        inner-color (first inner-ring)
        inner-ratio (* (dec level) cycle)
        inner (if axis?
                [[inner-color inner-ratio]]
                [[inner-color inner-ratio]
                 [inner-color
                  (mod
                   (+ inner-ratio along)
                   (count (last inner-ring)))]])
        adjacent (concat neighbors inner)
        outer? (< level (dec (count rings)))]
    (if outer?
      (let [outer-ring (nth rings (inc level))
            outer-color (first outer-ring)
            outer-ratio (* (inc level) cycle)
            outer-spaces (count (last outer-ring))
            outer (if axis?
                    (map
                     (fn [outer-step]
                       [outer-color (mod outer-step outer-spaces)])
                     [(dec outer-ratio)
                      outer-ratio
                      (inc outer-ratio)])
                    (let [along-outer (+ along outer-ratio)]
                      [[outer-color (mod along-outer outer-spaces)]
                       [outer-color (mod (inc along-outer) outer-spaces)]]))]
        [space (concat adjacent outer)])
      [space adjacent])))

(defn ring-adjacencies
  [rings level]
  (let [[color spaces] (nth rings level)
        num-spaces (count spaces)]
    (mapv
     (partial
      space-adjacencies
      rings
      level
      num-spaces)
     spaces)))

(defn find-adjacencies
  [rings]
  (let [[core-color core-spaces] (first rings)
        core (first core-spaces)
        adjacent {core (second (second rings))}
        others (mapv
                (partial ring-adjacencies rings)
                (rest rings))]
    (into adjacent others)))
