import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, Animated, FlatList } from "react-native";
import { colors } from "../theme/colors";
import AlertCard from "./components/AlertCard";
import SectionHeader from "./components/SectionHeader";

const mockCallAlerts = [
  {
    id: "1",
    resident: "Margaret Johnson",
    type: "Scammer claiming to be Tech Support",
    preview: "Caller claimed to be from Microsoft and asked for account access",
    risk: "High",
    timestamp: "5 min ago",
  },
  {
    id: "2",
    resident: "Robert Miller",
    type: "Suspicious IRS call",
    preview: "Caller threatened legal action if payment not made immediately",
    risk: "High",
    timestamp: "20 min ago",
  },
  {
    id: "3",
    resident: "Helen Davis",
    type: "Unknown caller",
    preview: "Caller refused to identify organization, requested bank details",
    risk: "Medium",
    timestamp: "45 min ago",
  },
];

export default function CallMonitoringScreen({ navigation }) {
  const [pulseAnim] = useState(new Animated.Value(1));
  const [isMonitoring] = useState(true);

  useEffect(() => {
    const startPulse = () => {
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.3,
          duration: 600,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 600,
          useNativeDriver: true,
        }),
      ]).start(() => startPulse());
    };

    if (isMonitoring) {
      startPulse();
    }
  }, [isMonitoring, pulseAnim]);

  return (
    <View style={styles.container}>
      <SectionHeader
        title="Call Monitoring"
        subtitle="Real-time fraud detection"
      />

      <View style={styles.statusCard}>
        <Text style={styles.statusLabel}>Monitoring Status</Text>
        <View style={styles.statusContent}>
          <Text style={styles.statusText}>
            {isMonitoring ? "MONITORING: ON" : "MONITORING: OFF"}
          </Text>
          <Animated.View
            style={[
              styles.pulsingDot,
              { transform: [{ scale: pulseAnim }] },
            ]}
          />
        </View>
      </View>

      <View style={styles.activeCallCard}>
        <Text style={styles.activeCallLabel}>Current Call</Text>
        <Text style={styles.residentName}>Eleanor White</Text>
        <Text style={styles.callDuration}>Duration: 3 minutes 22 seconds</Text>
        <View style={styles.riskIndicator}>
          <View style={styles.riskDot} />
          <Text style={styles.riskText}>Analyzing for suspicious patterns...</Text>
        </View>
      </View>

      <View style={styles.alertsSection}>
        <SectionHeader title="Recent Call Alerts" />
        <FlatList
          data={mockCallAlerts}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <AlertCard
              item={item}
              onPress={() => navigation.navigate("AlertDetails", { alert: item })}
            />
          )}
          scrollEnabled={false}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  statusCard: {
    backgroundColor: colors.pink,
    borderRadius: 14,
    padding: 20,
    marginBottom: 20,
  },
  statusLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 12,
    fontWeight: "bold",
  },
  statusContent: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  statusText: {
    fontSize: 22,
    fontWeight: "bold",
    color: colors.textDark,
    letterSpacing: 1,
  },
  pulsingDot: {
    width: 16,
    height: 16,
    borderRadius: 8,
    backgroundColor: "#51CF66",
    shadowColor: "#51CF66",
    shadowOpacity: 0.7,
    shadowRadius: 10,
    elevation: 5,
  },
  activeCallCard: {
    backgroundColor: "#F0F8FF",
    borderLeftWidth: 4,
    borderLeftColor: colors.green,
    borderRadius: 14,
    padding: 16,
    marginBottom: 20,
  },
  activeCallLabel: {
    fontSize: 12,
    color: "#999",
    marginBottom: 8,
    fontWeight: "bold",
  },
  residentName: {
    fontSize: 18,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 4,
  },
  callDuration: {
    fontSize: 13,
    color: "#666",
    marginBottom: 12,
  },
  riskIndicator: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  riskDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.green,
  },
  riskText: {
    fontSize: 13,
    color: colors.green,
    fontWeight: "500",
  },
  alertsSection: {
    marginBottom: 20,
  },
});
