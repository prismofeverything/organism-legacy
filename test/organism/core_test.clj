(ns organism.core-test
  (:require
   [clojure.test :refer :all]
   [organism.core :refer :all]))

(deftest build-rings-test
  (testing "building the rings"
    (let [rings (build-rings 5 [:yellow :red :blue :orange])]
      (println rings)
      (is (= 4 (count rings)))
      (is (= 31 (count (rings->spaces rings)))))))

(deftest space-adjacencies-test
  (testing "finding the adjacencies for a single space"
    (let [rings (build-rings 5 [:yellow :red :blue :orange :green])
          adjacent (space-adjacencies rings 3 15 [:orange 11])]
      (println "adjacent" adjacent)
      (is (= 6 (count (last adjacent)))))))

;; (deftest adjacencies-test
;;   (testing "finding adjacencies"
;;     (let [rings (build-rings 5 [:yellow :red :blue :orange])
;;           adjacencies (find-adjacencies rings)]
;;       (println adjacencies)
;;       (is (= 4 (count rings)))
;;       (is (= 31 (count (rings->spaces rings)))))))
