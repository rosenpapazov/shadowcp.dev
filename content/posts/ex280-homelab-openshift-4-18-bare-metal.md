+++
title = "An EX280 Home Lab on a Ryzen 7900: A Real 3-Node OpenShift 4.18 Cluster on Bare Metal"
slug = "ex280-homelab-openshift-4-18-bare-metal"
date = 2026-05-08T17:30:00+03:00
draft = false
description = "Building a production-shaped 3-node compact OpenShift 4.18 cluster on a Ryzen 7900 home lab. VLAN through a MikroTik router, Agent-based Installer, LVMS storage, and MetalLB."
images = ["/img/image.jpg"]
series = ["EX280 on Bare Metal"]
tags = ["openshift", "ex280", "kubernetes", "homelab", "kvm", "libvirt", "fedora", "mikrotik", "metallb", "lvms", "baremetal", "agent-installer"]

[params]
  seriesPart = 1
  seriesTotalParts = 1
+++

*A 3-node compact OpenShift 4.18 cluster on a Ryzen 7900 home lab, built for EX280 prep. VLAN 1303 through a MikroTik router, the Agent-based Installer, LVMS for dynamic storage, and MetalLB for real `LoadBalancer` services.*

---

## Why a real cluster?

The EX280 is a hands-on exam. You sit at a workstation, you get a cluster, and you're judged on whether you can make it do what the question asked, with configurations that survive a full reboot.

The temptation, when prepping at home, is to reach for **CRC** (CodeReady Containers). It boots in fifteen minutes, lives on a laptop, and is fine for some objectives (RBAC, NetworkPolicy, basic resource manifests).

But CRC papers over four areas that are exactly where most candidates lose points:

1. **MachineConfig rollouts.** A real cluster cordons a node, drains it, reboots it, and rejoins it. CRC is a single node, so the entire mechanism is invisible. When the exam asks you to push a `MachineConfig` and verify it landed, you want to have watched `oc get mcp -w` tick through `Updating → Updated` more than once before you sit down.
2. **etcd quorum.** Three members, real Raft, real failure modes. You can't intuit etcd backup and restore from a single-member cluster.
3. **Ingress with VIPs.** A bare-metal install runs `keepalived` with VRRP managing the API and Ingress VIPs. CRC doesn't have it.
4. **LoadBalancer services.** On bare metal you need MetalLB to make `type: LoadBalancer` actually mean something. CRC fakes it; the exam doesn't.

Given hardware that can run the real thing, this lab is a full **3-node compact cluster** on KVM, installed via the **Agent-based Installer**, with every control-plane node also acting as a worker. Red Hat documents the topology as a first-class production option.

## The hardware

A spare desktop:

- **CPU:** AMD Ryzen 9 7900 (12 cores / 24 threads, Zen 4)
- **RAM:** 64 GB DDR5
- **Storage:** 2× 2 TB NVMe
- **OS:** Fedora 43 on bare metal

The two NVMe drives are the part worth highlighting. One holds the host OS and VM boot disks. The second is consumed entirely by a libvirt storage pool that backs a second virtual disk on every VM. Those second disks become the physical volumes for **LVM Storage** (LVMS) inside the cluster, providing real dynamic PV provisioning.

That's the difference between "I read about LVMS" and "I have a default `StorageClass` that hands out PVs from a real device."

## Lab architecture

```
  ┌─ host: Fedora 43 (bare metal) ─────────────────────────────────┐
  │  Ryzen 9 7900, 64 GB, 2× 2 TB NVMe                             │
  │                                                                │
  │   libvirt + KVM                                                │
  │   ├── ocp-master-1   8 vCPU  16 GiB  120 GiB  (rendezvous)     │
  │   ├── ocp-master-2   8 vCPU  16 GiB  120 GiB                   │
  │   └── ocp-master-3   8 vCPU  16 GiB  120 GiB                   │
  │                                                                │
  │   Each VM also gets a 2nd disk (60 GiB) on nvme1 for LVMS      │
  │                                                                │
  │   libvirt network "ocplab" → VLAN 1303 bridge                  │
  │     Network: 10.130.3.0/24                                     │
  │     Gateway: 10.130.3.1 (MikroTik router)                      │
  │     Node IPs: 10.130.3.21, .22, .23                            │
  │     api.ocp4.lab          → 10.130.3.10  (apiVIP)              │
  │     *.apps.ocp4.lab       → 10.130.3.11  (ingressVIP)          │
  │                                                                │
  │   External services:                                           │
  │     DNS: MikroTik router at 10.130.3.1                         │
  └────────────────────────────────────────────────────────────────┘
```

Design choices worth calling out:

- **Compact cluster, not separate workers.** With 64 GB of host RAM, splitting into 3 control-plane plus N workers leaves no headroom for actual workloads. The compact topology (`controlPlane.replicas: 3` and `compute.replicas: 0`) tells the installer to make the masters schedulable. All three roles land on the same nodes.
- **VLAN 1303 on a bridged network, not libvirt NAT.** The cluster lives on a real subnet on my home network. DNS comes from my MikroTik router, not a host-local dnsmasq. This makes the cluster reachable from the rest of the LAN and forces me to operate it the way a production team would, with DNS as a real external dependency.
- **Static IPs with stable MACs.** No DHCP for cluster nodes. The Agent-based Installer wants the network pinned, and getting MACs and IPs right in `agent-config.yaml` is the most error-prone part of the whole install. Pinning them up front pays back tenfold.

| Component | vCPU | RAM | Storage |
|---|---|---|---|
| 3× control-plane VMs | 24 | 48 GiB | 3× 120 GiB OS + 3× 60 GiB data |
| Host (Fedora) + libvirt | residual | 16 GiB | rest of nvme0 |

| Device | Role |
|---|---|
| `/dev/nvme0n1` | Host OS + `/var/lib/libvirt/images` |
| `/dev/nvme1n1` | Second libvirt pool (`storage`), backs the LVMS disks inside the cluster |

## Stage 0: Host OS

Fedora 43, because it matches my daily driver. RHEL 9.5 with a no-cost Developer Subscription works identically and is closer to most employer environments.

During the OS install: `/` ~200 GiB, `/var/lib/libvirt/images` as a dedicated XFS partition of ~1.5 TiB, `nvme1n1` left untouched.

After first boot:

```bash
sudo dnf -y update
sudo dnf -y install @virtualization NetworkManager libvirt qemu-kvm \
    libvirt-daemon-config-network libvirt-daemon-kvm virt-install \
    libvirt-client virt-manager tar xz jq git wget htop \
    bash-completion nmstate

sudo systemctl enable --now libvirtd
sudo usermod -aG libvirt,kvm "$USER"
newgrp libvirt
```

Sanity-check that virtualization is actually exposed:

```bash
grep -c svm /proc/cpuinfo            # 24 expected on a 7900
virt-host-validate qemu | grep -E 'pass|fail|warn'
lsmod | grep kvm_amd
```

If `kvm_amd` isn't there, **SVM is disabled in the BIOS**. On Ryzen this lives under *Advanced → AMD CBS → CPU Common Options → SVM Mode = Enabled*. Some boards just call it *AMD-V*. Reboot, recheck.

## Stage 1: Disk layout for the second NVMe

The second drive becomes a libvirt storage pool. Each VM gets a 60 GiB disk from this pool that ends up as `/dev/vdb` inside the cluster, claimed by LVMS as a physical volume.

```bash
sudo wipefs -a /dev/nvme1n1
sudo parted /dev/nvme1n1 -- mklabel gpt
sudo parted /dev/nvme1n1 -- mkpart primary xfs 1MiB 100%
sudo mkfs.xfs -f -L ocp-pv /dev/nvme1n1p1

sudo mkdir -p /srv/ocp-storage
echo 'LABEL=ocp-pv /srv/ocp-storage xfs defaults,noatime 0 0' \
    | sudo tee -a /etc/fstab
sudo mount -a
```

Register it with libvirt:

```bash
sudo virsh pool-define-as storage dir --target /srv/ocp-storage
sudo virsh pool-start storage
sudo virsh pool-autostart storage
```

There are now two pools. `default` on `nvme0` for OS disks, `storage` on `nvme1` for the LVMS-backed second disks. The split matters: putting the data disks on a separate physical drive means cluster I/O doesn't trample the host's filesystem cache.

## Stage 2: VLAN 1303 on the MikroTik router

DNS and routing for the cluster live on the MikroTik. This is the bit that makes the lab feel like a real environment instead of a NAT bubble.

The assumption: a bridge (typically named `bridge`) already exists on the router with all physical ports as members, and `ether1` is the trunk port carrying VLAN 1303 to the lab host. Adjust interface names to taste.

**Add VLAN 1303 to the bridge and enable VLAN filtering:**

```routeros
/interface bridge vlan
add bridge=bridge tagged=ether1 vlan-ids=1303

/interface bridge
set bridge vlan-filtering=yes
```

**Create the VLAN interface and assign the gateway IP:**

```routeros
/interface vlan
add interface=bridge name=vlan1303 vlan-id=1303

/ip address
add address=10.130.3.1/24 interface=vlan1303 network=10.130.3.0
```

**Create a `LAB` interface list** so firewall rules can reference the lab subnet symbolically:

```routeros
/interface list
add name=LAB

/interface list member
add interface=vlan1303 list=LAB
```

**DNS static entries for the cluster.** `api.ocp4.lab` points at the apiVIP. `*.apps.ocp4.lab` is a regex, every Route hostname under the cluster's apps domain resolves to the ingressVIP:

```routeros
/ip dns static
add address=10.130.3.10 name=api.ocp4.lab
add address=10.130.3.10 name=api-int.ocp4.lab
add address=10.130.3.11 regexp=".*\\.apps\\.ocp4\\.lab"

/ip dns
set allow-remote-requests=yes
```

**Firewall rules** allowing DNS, ICMP, and NTP from the LAB list:

```routeros
/ip firewall filter
add action=accept chain=input comment="Allow DNS from LAB" \
    dst-port=53 in-interface-list=LAB protocol=udp
add action=accept chain=input comment="Allow DNS from LAB (TCP)" \
    dst-port=53 in-interface-list=LAB protocol=tcp
add action=accept chain=input comment="Allow ICMP from LAB" \
    in-interface-list=LAB protocol=icmp
add action=accept chain=input comment="Allow established/related" \
    connection-state=established,related
add action=accept chain=input comment="Allow NTP from LAB" \
    dst-port=123 in-interface-list=LAB protocol=udp
```

Verify before moving on:

```routeros
/interface bridge vlan print
/interface vlan print
/ip address print where interface=vlan1303
/interface list member print where list=LAB
/ip dns static print
/tool dns-lookup name=api.ocp4.lab        # should return 10.130.3.10
```

## Stage 3: Host networking (VLAN interface and bridge)

The lab host needs a bridged network on VLAN 1303 that libvirt can attach VMs to. NetworkManager handles this cleanly and survives reboots.

Identify your physical interface, whatever you have plugged into the trunk port:

```bash
ip link show
# Replace 'eno1' below with your actual interface name
PHYS_IF=eno1
```

**Create the VLAN interface, the bridge, and attach them:**

```bash
sudo nmcli connection add type vlan \
  con-name vlan1303 \
  ifname vlan1303 \
  dev ${PHYS_IF} \
  id 1303

sudo nmcli connection add type bridge \
  con-name br1303 \
  ifname br1303 \
  ipv4.addresses 10.130.3.254/24 \
  ipv4.gateway 10.130.3.1 \
  ipv4.dns 10.130.3.1 \
  ipv4.method manual

sudo nmcli connection modify vlan1303 \
  master br1303 \
  slave-type bridge

sudo nmcli connection up br1303
sudo nmcli connection up vlan1303
```

The host now sits on `10.130.3.254/24` with the MikroTik as both gateway and DNS resolver. Verify:

```bash
ip addr show br1303
ping -c 2 10.130.3.1
dig +short api.ocp4.lab          # should return 10.130.3.10
```

**Define the libvirt network in bridged mode.** No NAT, no internal libvirt DHCP, just an attachment to the host bridge:

```xml
<!-- /tmp/ocplab-net.xml -->
<network>
  <name>ocplab</name>
  <forward mode='bridge'/>
  <bridge name='br1303'/>
</network>
```

```bash
sudo virsh net-define /tmp/ocplab-net.xml
sudo virsh net-autostart ocplab
sudo virsh net-start ocplab
```

## Stage 4: OpenShift binaries

```bash
mkdir -p ~/ocp4 && cd ~/ocp4

# Pick the latest 4.18.z available. 4.18.13 was known-good when I built this
OCP_VER=4.18.13

curl -LO https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/${OCP_VER}/openshift-install-linux.tar.gz
curl -LO https://mirror.openshift.com/pub/openshift-v4/x86_64/clients/ocp/${OCP_VER}/openshift-client-linux.tar.gz

tar -xzf openshift-install-linux.tar.gz
tar -xzf openshift-client-linux.tar.gz

sudo install -m 0755 openshift-install /usr/local/bin/
sudo install -m 0755 oc kubectl /usr/local/bin/
oc version
openshift-install version
```

Pull secret from <https://console.redhat.com/openshift/install/pull-secret> goes in `~/ocp4/pull-secret.txt`. SSH key for break-glass access into the RHCOS nodes:

```bash
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519
```

## Stage 5: Cluster manifests

Two YAMLs feed the Agent-based Installer. Static IPs and stable MACs are the two details that have to be exactly right; the rest is autopilot.

```bash
mkdir -p ~/ocp4/cluster
cd ~/ocp4/cluster

PULL_SECRET=$(cat ~/ocp4/pull-secret.txt)
SSH_KEY=$(cat ~/.ssh/id_ed25519.pub)
```

### `install-config.yaml`

```bash
cat > install-config.yaml <<EOF
apiVersion: v1
baseDomain: lab
metadata:
  name: ocp4
compute:
- name: worker
  replicas: 0
controlPlane:
  name: master
  replicas: 3
networking:
  clusterNetwork:
  - cidr: 10.128.0.0/14
    hostPrefix: 23
  machineNetwork:
  - cidr: 10.130.3.0/24
  networkType: OVNKubernetes
  serviceNetwork:
  - 172.30.0.0/16
platform:
  baremetal:
    apiVIPs:
      - 10.130.3.10
    ingressVIPs:
      - 10.130.3.11
pullSecret: '${PULL_SECRET}'
sshKey: |
  ${SSH_KEY}
EOF
```

`compute.replicas: 0` plus `controlPlane.replicas: 3` is the explicit incantation for a 3-node compact cluster. The installer makes the masters schedulable for workloads automatically when it sees this combination.

### `agent-config.yaml`

The MAC addresses in this file are **arbitrary but must match what we assign to each VM**. That's how the agent matches a booting machine to its config:

```bash
cat > agent-config.yaml <<'EOF'
apiVersion: v1beta1
kind: AgentConfig
metadata:
  name: ocp4
rendezvousIP: 10.130.3.21
hosts:
  - hostname: ocp-master-1
    role: master
    interfaces:
      - name: enp1s0
        macAddress: 52:54:00:ab:cd:01
    networkConfig:
      interfaces:
        - name: enp1s0
          type: ethernet
          state: up
          ipv4:
            enabled: true
            address:
              - ip: 10.130.3.21
                prefix-length: 24
            dhcp: false
      dns-resolver:
        config:
          server:
            - 10.130.3.1
      routes:
        config:
          - destination: 0.0.0.0/0
            next-hop-address: 10.130.3.1
            next-hop-interface: enp1s0
  - hostname: ocp-master-2
    role: master
    interfaces:
      - name: enp1s0
        macAddress: 52:54:00:ab:cd:02
    networkConfig:
      interfaces:
        - name: enp1s0
          type: ethernet
          state: up
          ipv4:
            enabled: true
            address:
              - ip: 10.130.3.22
                prefix-length: 24
            dhcp: false
      dns-resolver:
        config:
          server:
            - 10.130.3.1
      routes:
        config:
          - destination: 0.0.0.0/0
            next-hop-address: 10.130.3.1
            next-hop-interface: enp1s0
  - hostname: ocp-master-3
    role: master
    interfaces:
      - name: enp1s0
        macAddress: 52:54:00:ab:cd:03
    networkConfig:
      interfaces:
        - name: enp1s0
          type: ethernet
          state: up
          ipv4:
            enabled: true
            address:
              - ip: 10.130.3.23
                prefix-length: 24
            dhcp: false
      dns-resolver:
        config:
          server:
            - 10.130.3.1
      routes:
        config:
          - destination: 0.0.0.0/0
            next-hop-address: 10.130.3.1
            next-hop-interface: enp1s0
EOF
```

### Generate the agent ISO

```bash
mkdir -p ~/ocp4/cluster-out
cp install-config.yaml agent-config.yaml ~/ocp4/cluster-out/

openshift-install --dir ~/ocp4/cluster-out agent create image
ls -l ~/ocp4/cluster-out/agent.x86_64.iso
```

The installer **consumes** the YAMLs. It deletes them after generating the ISO. Keep your source copies in `~/ocp4/cluster/` because you'll regenerate the ISO often during practice.

Move the ISO into the libvirt pool so it can be attached cleanly:

```bash
sudo cp ~/ocp4/cluster-out/agent.x86_64.iso /var/lib/libvirt/images/agent-ocp4.iso
sudo restorecon -v /var/lib/libvirt/images/agent-ocp4.iso 2>/dev/null || true
```

## Stage 6: Create the three VMs

```bash
#!/usr/bin/env bash
# ~/ocp4/create-vms.sh
set -euo pipefail

POOL_OS=/var/lib/libvirt/images
POOL_DATA=/srv/ocp-storage
ISO=${POOL_OS}/agent-ocp4.iso
NET=ocplab

declare -A MACS=(
  [ocp-master-1]=52:54:00:ab:cd:01
  [ocp-master-2]=52:54:00:ab:cd:02
  [ocp-master-3]=52:54:00:ab:cd:03
)

for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo qemu-img create -f qcow2 ${POOL_OS}/${vm}.qcow2 120G
  sudo qemu-img create -f qcow2 ${POOL_DATA}/${vm}-data.qcow2 60G

  sudo virt-install \
    --name ${vm} \
    --vcpus 8 \
    --memory 16384 \
    --cpu host-passthrough \
    --machine q35 \
    --boot uefi,hd,cdrom \
    --disk path=${POOL_OS}/${vm}.qcow2,bus=virtio,cache=none,io=native \
    --disk path=${POOL_DATA}/${vm}-data.qcow2,bus=virtio,cache=none,io=native \
    --disk path=${ISO},device=cdrom,bus=sata \
    --network network=${NET},mac=${MACS[$vm]},model=virtio \
    --osinfo detect=on,name=rhel9.4 \
    --noautoconsole \
    --import \
    --noreboot
done

for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh autostart ${vm}
  sudo virsh start ${vm}
done
```

`--cpu host-passthrough` lets the VMs see real Zen 4 features, which speeds up RHCOS workloads noticeably. CPU pinning isn't necessary for a lab. Setting `autostart` means the cluster comes back automatically after a host reboot.

## Stage 7: Wait for the install

The Agent-based Installer stages itself on the rendezvous host (master-1), validates the other two, writes RHCOS to disk on each, reboots them, and then bootstraps the control plane.

```bash
openshift-install --dir ~/ocp4/cluster-out agent wait-for bootstrap-complete --log-level=info
# expect: ~30 minutes

openshift-install --dir ~/ocp4/cluster-out agent wait-for install-complete --log-level=info
# expect: another 30-60 minutes for all operators to settle
```

Useful side-channel commands while you wait:

```bash
# Console of any VM
sudo virsh console ocp-master-1     # Ctrl-] to detach

# SSH directly to a node once RHCOS is on disk
ssh -i ~/.ssh/id_ed25519 core@10.130.3.21

# Once the API is up
export KUBECONFIG=~/ocp4/cluster-out/auth/kubeconfig
oc get nodes
oc get co
```

When `install-complete` returns, you'll see something like:

```
INFO Access the OpenShift web-console here: https://console-openshift-console.apps.ocp4.lab
INFO Login to the console with user: "kubeadmin", and password: "xxxxx-xxxxx-xxxxx-xxxxx"
```

Save that password into a password manager. The kubeconfig and auth files live in `~/ocp4/cluster-out/auth/`.

## Stage 8: The single most valuable step (snapshot)

After `install-complete`, before touching anything, snapshot every VM. You can rewind to a clean cluster in 30 seconds instead of 90 minutes.

```bash
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh snapshot-create-as --domain ${vm} \
       --name clean-install \
       --description "Pristine 4.18.x post-install" \
       --atomic
done
```

Reset to clean state at any time:

```bash
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh shutdown ${vm} || true
done
sleep 30
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh snapshot-revert ${vm} clean-install --running
done
```

Don't lean on snapshots for *every* exam objective. Some require persistence across a full cluster reboot, and you want practice fixing things properly. Use snapshots for cleanup *between* practice sessions, not as a substitute for getting the YAML right.

## Stage 9: Lab tooling on the host

```bash
sudo dnf -y install podman skopeo buildah jq yq make tree httpie

# Helm
HELM_VER=v3.17.0
curl -fsSL https://get.helm.sh/helm-${HELM_VER}-linux-amd64.tar.gz | tar -xz -C /tmp
sudo install -m 0755 /tmp/linux-amd64/helm /usr/local/bin/helm

# Kustomize
sudo dnf -y install kustomize

# oc bash completion + a couple of QoL bits
oc completion bash | sudo tee /etc/bash_completion.d/oc >/dev/null
echo 'alias k=oc' >> ~/.bashrc
echo 'complete -F __start_oc k' >> ~/.bashrc
echo 'export KUBECONFIG=~/ocp4/cluster-out/auth/kubeconfig' >> ~/.bashrc
```

## Stage 10: LVMS for dynamic storage

Each VM has a 60 GiB extra disk on `nvme1` that the cluster sees as `/dev/vdb`. The **LVM Storage** operator (LVMS, formerly ODF-LVM) turns those into a default `StorageClass`.

```bash
oc adm new-project openshift-storage

cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: openshift-storage-og
  namespace: openshift-storage
spec:
  targetNamespaces:
    - openshift-storage
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: lvms-operator
  namespace: openshift-storage
spec:
  channel: stable-4.18
  name: lvms-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
EOF

oc -n openshift-storage get csv -w     # wait for Succeeded
```

Then the `LVMCluster` CR that claims `/dev/vdb` on every node:

```bash
cat <<'EOF' | oc apply -f -
apiVersion: lvm.topolvm.io/v1alpha1
kind: LVMCluster
metadata:
  name: lvms-lab
  namespace: openshift-storage
spec:
  storage:
    deviceClasses:
      - name: vg1
        default: true
        deviceSelector:
          paths:
            - /dev/vdb
        thinPoolConfig:
          name: thin-pool-1
          sizePercent: 90
          overprovisionRatio: 10
EOF

oc get sc      # lvms-vg1 should appear marked (default)
```

Every PVC without an explicit `storageClassName` now goes to LVMS and lands on the second NVMe.

## Stage 11: MetalLB for real LoadBalancer services

The EX280 objectives include "Configure a load balancer service" under *Expose non-HTTP/SNI applications*. On bare metal that means MetalLB.

```bash
oc adm new-project metallb-system

cat <<'EOF' | oc apply -f -
apiVersion: operators.coreos.com/v1
kind: OperatorGroup
metadata:
  name: metallb-operator
  namespace: metallb-system
---
apiVersion: operators.coreos.com/v1alpha1
kind: Subscription
metadata:
  name: metallb-operator-sub
  namespace: metallb-system
spec:
  channel: stable
  name: metallb-operator
  source: redhat-operators
  sourceNamespace: openshift-marketplace
  installPlanApproval: Automatic
EOF

oc -n metallb-system get csv -w        # wait for Succeeded
```

Then the `MetalLB` CR plus an L2 address pool from the lab subnet:

```bash
cat <<'EOF' | oc apply -f -
apiVersion: metallb.io/v1beta1
kind: MetalLB
metadata:
  name: metallb
  namespace: metallb-system
spec: {}
---
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: lab-pool
  namespace: metallb-system
spec:
  addresses:
    - 10.130.3.100-10.130.3.200
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: l2adv
  namespace: metallb-system
spec:
  ipAddressPools:
    - lab-pool
EOF
```

Smoke test:

```bash
oc new-project lb-test
oc create deploy demo --image=registry.access.redhat.com/ubi9/httpd-24:latest
oc expose deploy/demo --port=8080 --type=LoadBalancer
oc get svc demo -w
# EXTERNAL-IP becomes 10.130.3.100 (or similar)
curl http://10.130.3.100:8080
```

That's a real `service.spec.type=LoadBalancer` answer to the exam objective, not a NodePort fudge.

## Stage 12: etcd backup

Not strictly an EX280 objective, but with a real 3-node etcd it would be silly not to practice the backup procedure:

```bash
oc debug node/ocp-master-1
# inside the toolbox shell:
chroot /host
sudo /usr/local/bin/cluster-backup.sh /home/core/backup
ls -la /home/core/backup
exit; exit

# Pull it back to the host
mkdir -p ~/ocp4/etcd-backups
scp -i ~/.ssh/id_ed25519 core@10.130.3.21:/home/core/backup/* ~/ocp4/etcd-backups/
```

Seeing the snapshot get taken on a real 3-node etcd is one of those things that converts knowledge into instinct.

## Daily lifecycle

```bash
# Stop cleanly (workloads first, then control plane)
for vm in ocp-master-3 ocp-master-2 ocp-master-1; do
  sudo virsh shutdown ${vm}
done

# Bring it back up (autostart handles host reboots; this is the manual path)
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh start ${vm}
done

# Reset to clean snapshot
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh shutdown ${vm} || true
done
sleep 30
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh snapshot-revert ${vm} clean-install --running
done

# Nuke and reinstall (when you want to rehearse the agent installer itself)
for vm in ocp-master-1 ocp-master-2 ocp-master-3; do
  sudo virsh destroy ${vm} 2>/dev/null || true
  sudo virsh undefine ${vm} --remove-all-storage --nvram 2>/dev/null || true
done
```

After every start, run `oc get co` until everything is `Available=True, Progressing=False, Degraded=False`. Don't start practising until the cluster is settled. You'll waste time chasing phantom failures otherwise.

## Verification checklist

```bash
# Network
ping -c 3 10.130.3.1
ping -c 3 10.130.3.21
dig +short api.ocp4.lab                              # 10.130.3.10
dig +short console-openshift-console.apps.ocp4.lab   # 10.130.3.11

# Cluster health
export KUBECONFIG=~/ocp4/cluster-out/auth/kubeconfig
oc get nodes                            # all Ready
oc get co                               # all Available, none Degraded
oc get clusterversion

# Storage and LB
oc get sc                               # lvms-vg1 is default
oc get pods -n openshift-storage        # all Running
oc get pods -n metallb-system           # all Running
oc get ipaddresspool -n metallb-system  # lab-pool 10.130.3.100-200

# Autostart
sudo virsh list --all --autostart       # all three VMs listed
```

## Things that bit me

A short list of issues that cost me time on the first build, in case they save you some on yours:

- **Cluster operators stuck `Progressing=True` after install.** Almost always DNS. Re-check that the host can resolve `api.ocp4.lab` *and* arbitrary names under `*.apps.ocp4.lab`, and that the nodes themselves use `10.130.3.1` as their resolver. The MikroTik regex match for `*.apps` is the most common typo.
- **MAC mismatch between `agent-config.yaml` and `virt-install`.** The agent ISO refuses to assign config to a host whose MAC it doesn't recognize, and the symptom is a node that boots, gets stuck on the agent service, and never registers. Triple-check both ends.
- **MetalLB pool overlapping the node range.** I originally picked `10.130.3.50-10.130.3.99` and accidentally collided with another LAN device. Keep the pool firmly inside an unused band. `100-200` here works.
- **Forgot to wait between practice sessions.** Snapshot-revert leaves the cluster in `Progressing=True` for several minutes while operators reconcile. If you start applying manifests immediately, half of them race the rollout and fail confusingly.

## What this lab buys you

A real 3-node etcd quorum. Real `MachineConfig` rollouts you can watch cordon, drain, reboot, and rejoin a node at a time. Real ingress with VIPs that actually fail over. A default `StorageClass` backed by a real device. Real `LoadBalancer` services. DNS as a real external dependency, not a host-local hack.

In other words: roughly the same surface area you'll see on the exam, with none of the fictions CRC papers over.

## What's next

The next posts in this series walk through specific EX280 objectives using this lab as the harness. Authentication and authorization (HTPasswd identity providers, groups, RBAC, project request templates). Network security (NetworkPolicy, ingress controllers, route types). Operators and the OperatorHub (installing, pinning, uninstalling cleanly). Application security (ServiceAccounts, SCCs, Jobs and CronJobs). Resource management (quotas, LimitRanges, ClusterResourceQuota). Each objective broken down into a scenario, the YAML that solves it, verification steps, and the common mistakes that trip people up.

If you've prepped for an EX-class exam before, you know the difference between reading the docs and having muscle memory. This lab exists to manufacture muscle memory.

---

*Series: EX280 on Bare Metal*
- **Part 1: Building the lab on a Ryzen 7900** (you are here)
